# Vectorless RAG

Multi-user question answering over PDFs **without a vector database**. Documents are indexed as a
page tree with [PageIndex](https://github.com/VectifyAI/PageIndex) (local mode), figures are described
by a vision model and written into the PDF as invisible text, and a LangChain agent searches the
tree, looks at page images itself when exact values matter, and answers with page-level citations.
Each user only ever sees their own documents.

## How it works

```
upload ─► API ─► check, deduplicate, save original ─► registry (queued)
                                                          │
                                         ingestion worker (in the same process)
                                                          │
          1. find figure pages (placed images, low-text drawing clusters)
          2. describe each figure page with a vision model, store the description
          3. write the descriptions into a copy of the PDF as invisible text
          4. index that copy with the user's own PageIndex library
                                                          │
                                                registry (completed / failed)

question ─► API ─► LangChain agent (OpenAI Responses API)
                   tools: 4 read-only PageIndex tools + view_pages (original page images)
                   ─► answer with citations, mapped to our document ids
```

- **No embeddings, no chunking.** PageIndex builds a tree of sections with summaries; the agent
  navigates it like a person reading a table of contents.
- **Figures are searchable.** PageIndex reads only the text layer, so each figure's description is
  added to that layer, marked `[FIGURE DESCRIPTION pN fig1]`.
- **Figures are readable.** `view_pages` hands the original page images to the answering model,
  which reads exact values itself instead of trusting the description.
- **Isolation by construction.** Every user has their own PageIndex library on disk, and the
  agent's tools are bound to it.

The full design is in [`docs/vectorless-rag-spec.md`](docs/vectorless-rag-spec.md); the experiments
behind it are in [`docs/spike-findings.md`](docs/spike-findings.md).

## Quick start

Requires Python 3.13 and an OpenAI API key. Commands are for Windows (Git Bash); on Linux or macOS
use `.venv/bin/` instead of `.venv/Scripts/`.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
cp .env.example .env          # then set OPENAI_API_KEY; every setting is listed there with its default
.venv/Scripts/python -m vectorless_rag.api
```

The service listens on `http://127.0.0.1:8000`; interactive docs are at `/docs`. Uploaded files,
enriched copies, PageIndex libraries and the SQLite registry go to `./var` (`DATA_ROOT`).

## Using the API

Authentication is out of scope: an upstream layer is expected to send the trusted user id in the
`X-User-Id` header on every request.

```bash
# upload a PDF: 202 + the queued document (200 + the existing one for a file uploaded before)
curl -H "X-User-Id: alice" -F "file=@manual.pdf" http://127.0.0.1:8000/documents

# watch it go queued -> enriching -> indexing -> completed
curl -H "X-User-Id: alice" http://127.0.0.1:8000/documents/<id>

# ask about all of your documents, or pass "document_ids" to pick some
curl -H "X-User-Id: alice" -H "Content-Type: application/json" \
     -d '{"question": "At what temperature does the device shut itself off?"}' \
     http://127.0.0.1:8000/query
```

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/documents` | Upload a PDF (multipart field `file`) |
| `GET` | `/documents` | The user's documents, newest first |
| `GET` | `/documents/{id}` | One document, with its status and figure count |
| `GET` | `/documents/{id}/figures` | The stored figure descriptions |
| `DELETE` | `/documents/{id}` | Delete from PageIndex, the registry and disk |
| `POST` | `/query` | Ask a question: `question`, optional `document_ids`, `history` and `detail` |

An answer looks like this; `[1]` points at the first citation, and `stats` sums up how the agent got there:

```json
{
  "answer": "It shuts itself off at 60 °C (140 °F). [1]",
  "citations": [{"index": 1, "document_id": "…", "filename": "manual.pdf", "page": 27, "from_figure": false}],
  "trace_id": "…",
  "stats": {"model_calls": 3, "tool_calls": 2, "pages_read": 1, "images_viewed": 0,
            "input_tokens": 18234, "output_tokens": 412, "reasoning_tokens": 256, "duration_ms": 9120}
}
```

With `"detail": "full"` the answer also lists `steps`: every model call (tokens, the tools it asked for)
and every tool call (arguments, pages read, `ok` / `error` / `blocked`, the first 2,000 characters of
the result) in order. Page images show only as `[page image: manual.pdf p30]`.

Errors are `{"detail": "..."}` with `401` (no user header), `404` (no such document for this user),
`409` (document not indexed yet), `413` (file too large or too many pages), `415` (not a readable
PDF) or `504` (the agent ran out of steps or time).

## Configuration

Every setting is an environment variable (or a line in `.env`); [`.env.example`](.env.example) lists
them all with their defaults. The main ones:

| Setting | Default | What it does |
|---|---|---|
| `OPENAI_API_KEY` | required | Used for indexing, figure descriptions and answering |
| `CHAT_MODEL` | `gpt-5.6-sol` | Answers questions; must read images and call tools |
| `VISION_MODEL` | `gpt-5.6-luna` | Describes figures at upload time |
| `INDEX_MODEL` | `gpt-5.6-luna` | Builds the PageIndex tree and summaries |
| `DATA_ROOT` | `./var` | Uploads, enriched copies, PageIndex libraries |
| `DATABASE_URL` | `sqlite:///./var/app.db` | The registry (Postgres in production) |
| `MAX_UPLOAD_MB` / `MAX_PAGES` | `50` / `500` | Upload limits |
| `AGENT_MAX_STEPS` / `AGENT_TIMEOUT_S` | `20` / `120` | Limits per question |
| `AGENT_REASONING_SUMMARY` | `auto` | `off`, `auto` or `detailed`: the model's reasoning summaries in the full `/query` steps |

## Development

```bash
.venv/Scripts/pyright                                      # type check src, tests and evals: 0 errors expected
.venv/Scripts/python -m pytest                             # unit and integration tests, no network
RUN_LIVE_TESTS=1 .venv/Scripts/python -m pytest tests/integration   # also the tests that call OpenAI (cents)
.venv/Scripts/python -m evals.run                          # the eval: 24 questions, about $2-3 (calls OpenAI)
```

The live tests include the required image-delivery check: a question only a page image can answer,
plus a control with a blank image where the agent must say it cannot see the answer.

The eval (`evals/`) asks the questions in `evals/questions.yaml` through the real service, without
choosing documents, and scores how the agent found pages and how good its answers are; an LLM judge
(`EVAL_JUDGE_MODEL`, default `gpt-5.6-sol`) grades the answers. The first run builds the eval library
from `Data/` under `evals/var/`; later runs reuse it. `--only ID ...` asks some questions, `--repeat N`
asks each one N times. Each run writes `results.jsonl` and `summary.md` to `evals/results/<date>/`.

```
src/vectorless_rag/
  api/          FastAPI app, routes, error mapping; composition root
  worker/       in-process ingestion worker and per-user locks
  operations/   use cases (documents, ingestion, questions) and the ports they depend on
  db/ pdf/ vision/ indexing/ agent/ storage/   adapters behind those ports
  models/       shared Pydantic records
tests/          fakes/ (in-memory ports), unit/, integration/
evals/          the eval set, runner, metrics, judge and report; results/ holds past runs
docs/           spec and spike findings
spikes/         the throwaway experiments behind the spec
Data/           sample PDFs used as read-only test input
```

## Limits (v1)

- PDF only; fully scanned documents are rejected with a clear message.
- Figure descriptions use a Latin-1 font, so non-Latin text in them (e.g. CJK) is dropped.
- One service process: the per-user indexing lock is in-process.
- No streaming answers yet.

The sample PDFs in `Data/` are Tobii Dynavox user manuals, kept only as test input.
