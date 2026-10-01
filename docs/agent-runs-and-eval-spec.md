# Agent runs and evaluation — Spec (v1.0, locked)

Status: locked on 2026-10-01 (decisions in Section 7); built on branch `feature/agent-runs-and-eval`.
Extends the main spec
(`docs/vectorless-rag-spec.md`, v0.4); Section 6 lists what changes there once this is built.

Three parts:
- **A. Agent run record:** every question produces a record of how the agent answered. The API returns
  a brief response by default and the full steps on request.
- **B. Langfuse later:** the seams that make adding Langfuse a small change. Not built now.
- **C. Evaluation:** an eval set over the sample manuals, a runner, and retrieval and generation metrics.

---

## 1. Why

Today `AnswerAgent.answer()` returns only the final text, so nobody can see which pages the agent
read, which tools it called, what it cost, or why it answered as it did. The eval needs exactly
that, and so will debugging and Langfuse. One record serves all three.

---

## 2. Part A — the agent run record

### 2.1 What is recorded

One `AgentRun` per question (new `models/runs.py`):

| Field | Meaning |
|---|---|
| `answer` | The final text, with `<cite>` tags (the use case resolves them as today) |
| `steps` | Everything the agent did, in order (below) |
| `totals` | Model calls, tool calls, unique pages read, page images viewed, input / output / reasoning tokens, duration |

Each step is either a model call or a tool call.

**Model step**

| Field | Meaning |
|---|---|
| `index` | Position in the run, from 1 |
| `duration_ms` | Time of the call |
| `input_tokens`, `output_tokens`, `reasoning_tokens` | From `AIMessage.usage_metadata` |
| `reasoning_summary` | The model's reasoning summaries, if available (2.3); empty otherwise |
| `text` | Any text the model wrote in this call (the final answer is the last one) |
| `tool_calls` | Names of the tools it asked for |

**Tool step**

| Field | Meaning |
|---|---|
| `index` | Position in the run |
| `tool` | `browse_documents`, `get_document`, `get_document_structure`, `get_page_content`, `view_pages` |
| `arguments` | The arguments, as sent |
| `document` | The document name it targets, if any |
| `pages` | The pages it read (`get_page_content`, `view_pages`), as numbers |
| `outcome` | `ok`, `error` (the tool reported an error, e.g. unknown name, bad page spec) or `blocked` (a limit stopped it) |
| `result_preview` | The start of the result (up to 2,000 characters). Page images appear as `[page image: <name> p12]`, never as bytes |
| `duration_ms` | Time of the call |

### 2.2 How it is recorded

- A recording middleware in `agent/` (`wrap_model_call` and `wrap_tool_call`) times each call and
  appends a step to a recorder that belongs to one question. The agent is already built per question,
  so the recorder is too.
- Tokens come from each `AIMessage.usage_metadata`. Pages are the pages the agent actually got, in the
  page-spec format of `view_pages` (`"12"`, `"12,13"`, `"12-13"`): for `view_pages`, the pages in its
  arguments (it shows all of them or rejects the call); for `get_page_content`, the `returned_pages`
  PageIndex reports, which leave out pages past the end of the document and pages cut to keep the
  reply under its size limit (decision 6). The arguments stay in the step as sent.
- A tool result is `error` when PageIndex returns `{"error": ...}` or `view_pages` returns its
  rejection text; `blocked` when a `ToolCallLimitMiddleware` stopped the call.
- The run is recorded even when it ends in `AnswerIncomplete`: the error carries the partial run, so the
  eval and the logs still see what happened.

### 2.3 Reasoning summaries

OpenAI does not expose its models' raw reasoning. Through the Responses API it can return **reasoning
summaries**, which LangChain delivers as `reasoning` content blocks (LangChain docs, "Standard content
blocks"). New setting:

| Name | Purpose | Default |
|---|---|---|
| `AGENT_REASONING_SUMMARY` | `off`, `auto` or `detailed`: ask the answering model for reasoning summaries | `auto` |

To check first (build step 3): whether `gpt-5.6-sol` returns summaries on our key, and what they cost.
The default stays `auto` only if they work; otherwise `off`.

Checked on 2026-10-01 (`docs/spike-findings.md`, "Agent-runs step 3"): they work, with no visible extra
cost; a model call that barely reasons gets none. The default is `auto`.

### 2.4 Port change

```python
class AnswerAgent(Protocol):
    def answer(self, instructions, tools, view_pages, messages, labels: RunLabels) -> AgentRun: ...
```

- Returns `AgentRun` instead of `str`.
- `RunLabels` (new): `trace_id` and `user_key`. `QuestionAnswering` creates the `trace_id` **before**
  the run (today it is made afterwards), so logs, the API response and later Langfuse all share it.

### 2.5 API: brief or full

`POST /query` gets one optional field:

```json
{"question": "...", "document_ids": [], "history": [], "detail": "brief"}
```

- `"brief"` (default): today's response plus a small `stats` block (tool calls, pages read, page images
  viewed, tokens, duration).
- `"full"`: the same, plus `steps` (2.1).

```json
{
  "answer": "It shuts itself off at 60 °C (140 °F). [1]",
  "citations": [...],
  "trace_id": "...",
  "stats": {"model_calls": 4, "tool_calls": 3, "pages_read": 2, "images_viewed": 0,
            "input_tokens": 18234, "output_tokens": 412, "reasoning_tokens": 256, "duration_ms": 9120},
  "steps": [
    {"index": 1, "kind": "model", "duration_ms": 1800, "tool_calls": ["browse_documents"], "reasoning_summary": ["..."]},
    {"index": 2, "kind": "tool", "tool": "browse_documents", "arguments": {}, "outcome": "ok", "result_preview": "..."}
  ]
}
```

What never leaves the service: page image bytes, the system prompt, and tool results past the
2,000-character preview. Steps only ever describe the asking user's own run.

### 2.6 Storage and logs

- Runs are not stored in v1; the response carries them. Langfuse stores them later (Part B).
- One log line per question with the `trace_id` and the totals, never the question or document text.

---

## 3. Part B — Langfuse later (seams only)

Built in a later branch. Checked in the Langfuse docs: LangChain is traced with
`langfuse.langchain.CallbackHandler`, passed in the run's `config` together with metadata
(`langfuse_user_id`, `langfuse_session_id`, `langfuse_tags`); a trace id from our own system can be
used through `Langfuse.create_trace_id(seed=...)` and `trace_context`.

What this branch prepares:
- `LangChainAnswerAgent` accepts an optional list of LangChain callback handlers from the composition
  root and passes them with the run's `config`. Empty for now.
- `RunLabels` (2.4) carries what Langfuse needs: `trace_id` (seed for the Langfuse trace id) and
  `user_key` (sent as the Langfuse user id, so raw user ids never leave the service).

What the Langfuse branch will add:
- Settings `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`; tracing is off when they are
  unset. The keys are passed to the Langfuse client explicitly, like the OpenAI key (CLAUDE.md).
- Optionally, ingestion traces (vision calls, PageIndex's LiteLLM calls) and eval scores sent to Langfuse.

---

## 4. Part C — evaluation

### 4.1 The eval set

A reviewed file of questions over the 4 sample manuals in `Data/`: `evals/questions.yaml`.

```yaml
- id: tdi110-shutoff-temperature
  question: At what temperature does the TD I-110 shut itself off to avoid harm?
  type: text                      # text | figure | multi_page | cross_document | unanswerable
  gold:                           # where the answer is (1-based pages of the original PDF)
    - document: TDI-110_UsersManual_en-US_WEB_1000958-01.pdf
      pages: [27]
  figure_pages: []                # pages whose image is needed to answer (figure questions)
  answer: It shuts itself off at 60 °C (140 °F).
  key_facts: ["60", "140"]        # a correct answer must contain each of these
  evidence: p27, "...", the safety section text
```

| Type | Count | What it tests |
|---|---|---|
| `text` | 8 | The answer is in the text layer |
| `figure` | 8 | The answer is only in a screenshot, photo or drawing (at least a third of the set, main spec 13) |
| `multi_page` | 3 | The answer needs two or more pages |
| `cross_document` | 2 | The answer needs two manuals (e.g. compare two devices) |
| `unanswerable` | 3 | The manuals do not contain the answer; the right answer is to say so |
| **total** | **24** | Spread over all 4 manuals |

How the ground truth is made: I read the page text and look at the rendered pages, write each
question with its answer, gold pages and evidence, and you review the file before anything runs.

### 4.2 The run

- One eval user's library holds all 4 manuals, built with the normal ingestion pipeline (figure
  descriptions included). Built once and reused, under `evals/var/` (not committed).
- Each question is asked **without** `document_ids`, so the agent has to find the right manual; that is
  what the document hit rate measures. One run per question by default (`--repeat N` for more).
- Models: the indexing, vision and chat models come from the settings (`.env`), as in the service.
- Not in this round: a `plain` library without figure descriptions, to show what enrichment adds
  (main spec 13). It can be added later as a variant of the same runner.
- Command: `python -m evals.run [--only ID ...]`. Output in `evals/results/<date>/`: `results.jsonl`
  (one line per question: run, scores, judge reasons) and `summary.md`. Both are committed.

### 4.3 Retrieval metrics

"Read" means the agent called `get_page_content` or `view_pages` on that page. "Opened" a document means
any tool call targeted it. Per question, then averaged; reported overall and per type. Unanswerable
questions have no gold pages: they count only for navigation cost.

| Metric | Definition |
|---|---|
| **Page recall** | gold pages read ÷ gold pages |
| **Page precision** | gold pages read ÷ unique pages read |
| **Hit@3** | a gold page is among the first 3 unique pages read |
| **MRR** | 1 ÷ position of the first gold page among the unique pages, in reading order (0 if none) |
| **Document hit rate** | the agent opened a gold document; also reported: the first document it opened was gold |
| **Figure hit rate** | figure questions only: `view_pages` was called on a gold figure page |
| **Navigation cost** | tool calls, unique pages read, page images viewed, model calls, tokens, duration and cost (from a price table in the eval config) |
| **Dead ends** | tool calls that ended `error` or `blocked`, plus page reads that contained no gold page |

### 4.4 Generation metrics

| Metric | How it is measured |
|---|---|
| **Answer correctness** | Two scores: key facts (all `key_facts` appear in the answer, after normalising case, spacing and `°`) and a judge grade against the reference answer: correct 1, partly 0.5, wrong 0 |
| **Faithfulness** | Judge: share of the answer's factual claims that are supported by what the agent saw (the text of the pages it read, the images of the pages it viewed) |
| **Citation accuracy** | Citation precision (cited pages that are gold ÷ cited pages) and citation recall (gold pages cited ÷ gold pages) |
| **Answer relevance** | Judge: does the answer address the question that was asked: yes 1, partly 0.5, no 0 |
| **Refusal accuracy** | Unanswerable questions: the agent says the documents do not contain it and invents nothing (judge). Answerable questions: the false-refusal rate |

The judge:
- `EVAL_JUDGE_MODEL` (default `gpt-5.6-sol`) through the Responses API, one call per question returning JSON with each judged score
  and a one-line reason. It sees the question, the reference answer, the agent's answer, and the evidence
  the agent saw (page text it read, images of pages it viewed).
- Its reasons are stored, and you spot-check about 5 questions per run, because a judge can be wrong too.

### 4.5 Report

`summary.md` holds:
- one table of every metric, overall and per question type;
- a per-question table with pass and fail marks;
- the failures with the judge's reasons;
- cost and time.

### 4.6 Cost and time (estimate)

- Building the library once: about $0.20 (180 pages, 60 figure pages described).
- Per run of 24 questions: answering about $0.05–0.10 each, judging about $0.02 each, so **about $2–3**
  and 15–25 minutes.

### 4.7 Where the code goes

```
evals/                 a development tool next to the service, not part of it
  questions.yaml       the reviewed eval set
  questions.py         loads and checks it
  config.py            EVAL_JUDGE_MODEL and the price table
  run.py               entry point: builds the library, asks the questions, writes results
  metrics.py           pure functions: (question, AgentRun) -> scores
  judge.py             the LLM judge
  report.py            summary.md
tests/unit/test_eval_metrics.py   the metric functions on hand-made runs, and the eval set file
tests/unit/test_eval_judge.py     the judge against a fake HTTP layer
tests/unit/test_eval_run.py       the runner and report on in-memory fakes
```

As built (step 5): a key fact may list alternatives as `"a|b"`, and key facts ignore all spaces as
well as case, `°` and curly quotes, so "60 °C" and "60°C" match. A run that ends without an answer is
not judged and scores 0 for correctness and relevance. A judge failure leaves the judged scores empty
and the run goes on. A day's second run writes to `<date>-2/`, so nothing is overwritten.

The runner reuses the service's own pieces (ingestion, `QuestionAnswering`, `detail="full"`), so it
measures the real system.

---

## 5. Build order on this branch

1. Run record: models, recording middleware, port change, use case (unit tests on the scripted model).
2. API: `detail`, `stats` and `steps`.
3. Reasoning summaries: check them on our key; then the setting.
4. Eval set: `evals/questions.yaml`, then **your review**.
5. Eval runner, metrics and judge, with unit tests for the metrics.
6. Build the library, run, report, and discuss the results.
7. Langfuse: a later branch (Part B).

---

## 6. Changes to the main spec once built

| Section | Change |
|---|---|
| 5.6 | `AnswerAgent` returns `AgentRun`; recording middleware; `RunLabels` |
| 7 | `detail` on `/query`; `stats` and `steps` in the response |
| 10 | `AGENT_REASONING_SUMMARY` (later `LANGFUSE_*`) |
| 12 | Observability: the run record now; Langfuse next |
| 13 | The eval set and metrics are defined here |
| 18 | `models/runs.py`, the recording middleware, `evals/` |

---

## 7. Decisions (2026-10-01)

1. Reasoning summaries: checked on our key in build step 3; `AGENT_REASONING_SUMMARY` defaults to
   `auto` if they work, `off` otherwise.
2. Judge model: `gpt-5.6-sol`, as the default of a new setting `EVAL_JUDGE_MODEL`. The indexing,
   vision and chat models come from `.env`, as in the service.
3. No with/without-enrichment comparison in this round; only the agreed workflow (Section 5).
4. Results are committed: `results.jsonl` and `summary.md` under `evals/results/<date>/`.
5. The definitions of dead ends and of "opened a document" (4.3) are confirmed.
6. (2026-10-01, during build step 1) A page counts as read only if the agent actually got it: for
   `get_page_content`, PageIndex's `returned_pages`, not the pages asked for (2.2). Otherwise a reply
   cut for size would count pages the agent never saw, and page recall would be too high.
