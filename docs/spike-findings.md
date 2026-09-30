# Spike findings

Date: 2026-09-29. Spikes live in `spikes/` (throwaway, typed, pyright-clean). Raw outputs are in
`spikes/out/*/report.json`. These records ground spec v0.4 (`docs/vectorless-rag-spec.md`).

## Environment

- Windows 11, Python 3.13.5 (Anaconda base) → `spikes/.venv`. `OPENAI_API_KEY` read from `.env`, never printed.
- Installed (pinned in `spikes/requirements.txt`): `pageindex==0.2.20`, `langchain==1.4.3`,
  `langchain-core==1.6.5`, `langchain-openai==1.6.6`, `openai==2.54.0`, `pymupdf==1.28.2`,
  `python-dotenv==1.2.3`. Pulled in: `langgraph 1.2.12`, `litellm 1.103.0`, `openai-agents 0.20.0`,
  `PyPDF2 3.0.1`, `pypdfium2 5.13.0`, `pydantic 2.13.5`.
- **`openai==3.20.0` (latest) cannot be installed next to `pageindex`**: its dependencies (litellm /
  openai-agents) cap `openai` below 3. pip resolved `2.54.0`.
- `pyrightconfig.json` at the repo root points pyright at `spikes/.venv`. The pyright CLI resolves all
  imports (0 errors). The editor's pyright LSP started before the config existed and kept showing
  "import could not be resolved" until Claude Code is restarted.
- S0 (`spikes/s0_models.py`): `gpt-5.6-luna` and `gpt-5.6-sol` exist on the key; `gpt-5.6-terra` too.
- Prices read from the OpenAI pricing page on 2026-09-29 (per 1M tokens, input / output):
  sol $4 / $20, terra $2 / $12, luna $0.20 / $1.20.
- Total API spend for all spikes: well under $1 (token counts below).

## Survey 0 — the sample PDFs (read-only)

All four files in `Data/` are Tobii Dynavox device user manuals (SCHEMA ST4), 595×792 pt pages.
`Data/` was only read, never written.

| File | Pages | Text layer | Embedded bookmarks | Scanned pages |
|---|---|---|---|---|
| TD I-Series I-13/I-16 | 65 | every page (median ~1.6k chars) | 23 top-level | none |
| TD Navio | 33 | every page (cover 37 chars) | 24 | none |
| TDI-110 | 32 | every page (cover 22 chars) | 22 | none |
| TD Pilot | 50 | every page (cover 29 chars) | 25 | none |

- **No data charts anywhere.** Figures are product photos, app screenshots, line drawings with
  numbered callouts, and ruled tables (spec, EMC, symbol tables).
- Callout numbers are in the text layer but scrambled (`"122 3456 9 78"`); the Position/Description
  tables are text. What figures add is mostly *where* something is, or *what a screen shows*.
- Running header/footer text on every page.

## Spike A — invisible text reaches PageIndex (Open Q2, Q3)

**Ran:** `spikes/spike_a_invisible_text.py`. Copied TDI-110, wrote hand-written descriptions with
canary words as invisible text (PyMuPDF `insert_textbox(..., fontsize=6, fontname="helv",
render_mode=3)`, inside the figure's bounding box) on p13 (raster photo) and p14 (vector drawing).
Saved the copy under the **original file name** in its own folder. Then: free checks, and one real
local index with `gpt-5.6-luna`.

**Result:**
- Both pages render pixel-identical to the original at 100 DPI.
- PageIndex local mode uses **two extractors**: PyPDF2 `extract_text()` for stored page text
  (`get_page_content`, node text) and pdfium (Flash) for the tree and summaries. **Both** contain the
  canaries on both pages.
- Flash tree, original vs enriched: identical — 124 nodes from bookmarks, and 62 nodes from layout
  detection with bookmarks off (`use_embedded_toc=False`). No title contains the inserted text.
- Real index: 32 pages, 32.6 s, 29 LLM calls, 39,446 input + 7,915 output tokens (≈ $0.02).
  Stored name = original file name. `get_page_content(doc, "13-14")` contains both canaries.
- Summaries: the canary words do not appear (summaries paraphrase). The p13 photo description *was*
  absorbed ("The figure shows its black rounded tablet body, wide bezel, top sensors, printed model
  label, and bottom feet"). The p14 drawing description was **not**: p13–16 became one leaf node
  whose ~150-word summary spent its words on body text.
- PyPDF2 glued the marker to the previous line: `"7 Microphone[FIGURE DESCRIPTION p14 fig1] ..."`.
- `summary_max_words`, `summary_concurrency` (default 64 parallel calls), `use_embedded_toc` and
  `optimize` can be set in the client's `index={...}` dict.

**Conclusion:** Q2 — yes, invisible text is picked up by both extraction paths; no fallback needed.
Q3 — no effect on structure detection (with or without bookmarks). Descriptions always reach page
content; reaching a node summary is likely but not guaranteed on multi-page leaf nodes. Start each
description on its own line.

## Spike B — page images reach the main agent (Open Q1, Q9)

**Ran:** `spikes/spike_b_view_pages.py`. `create_agent(ChatOpenAI("gpt-5.6-sol"), tools=[view_pages])`,
where `view_pages(doc_name, pages)` renders pages of the **original** PDF with PyMuPDF at 170 DPI (PNG)
and returns content blocks `[text label, image]`. Question answerable only from the image (I-Series
p30 calibration screenshot; its labels are not in the text layer): how many points are "Great" and
where is "No data"? Truth: 3, top centre.

| Variant | Result |
|---|---|
| Image blocks in `ToolMessage`, **Responses API** (`use_responses_api=True`) | **Correct.** ~3.2k input tokens for the image |
| Same, Chat Completions | **400 error**: gpt-5.6-sol does not allow function tools with reasoning on `/v1/chat/completions` |
| Chat Completions with `reasoning_effort="none"` | Call succeeded, answer **wrong** ("4 Great, lower-right"); only 352 input tokens → image silently dropped |
| Control: tool returns text only ("attached below") | Answer **invented** ("8 Great, upper right") |
| Fallback: text tool result + middleware adds images as a user message after it | Correct |
| Trim: two `view_pages` calls, middleware keeps latest image set only | Images sent per model call: 0, 1, 1. Both answers correct |
| `image_url` block with `detail: "high"` | Correct, ~3.0k image tokens |
| `image_url` block with `detail: "low"` | ~330 image tokens, location **wrong** |

- LangChain's standard `{"type": "image", "base64": ...}` block **drops** OpenAI's `detail` (with or
  without `extras`). An OpenAI-format `image_url` (or `input_image`) block keeps it.
- Limits: `ModelCallLimitMiddleware(run_limit=...)` and `ToolCallLimitMiddleware(tool_name=..., run_limit=...)`
  exist in `langchain.agents.middleware` and were used. Image trimming is a small custom
  `@wrap_model_call` middleware that rewrites only the messages sent to the model (agent state untouched).

**Conclusion:** Q1 — the preferred mechanism in spec 5.7 works: image blocks inside the tool result,
**only through the Responses API**. The fallback also works but is not needed. Failure is silent (the
model invents an answer), so the system prompt must say "if you cannot see a page image, say so", and
the image-delivery check must be an automated test. Use `image_url` blocks with `detail="high"`.
Q9 — `create_agent` + the two limit middlewares + a custom `wrap_model_call` trimmer.

## Spike C — PageIndex tools in LangChain, isolation (Open Q5, Q6, Q7)

**Ran:** `spikes/spike_c_tools_isolation.py`. User A = Spike A's storage (TDI-110). User B = a fresh
`storage_path` with two 2-page PDFs cut from Navio and TD Pilot.

**Result:**
- `client.agent_tools()` (read-only) returns 4 plain functions with real signatures and Google-style
  docstrings:
  `browse_documents(offset: int = 0, limit: float = 10)`,
  `get_document(doc_name: str, wait_for_completion: bool = False)`,
  `get_document_structure(doc_name: str, part: int = 1, wait_for_completion: bool = False)`,
  `get_page_content(doc_name: str, pages: str, wait_for_completion: bool = False)`. All return JSON
  strings and never raise; errors come back as `{"error": ...}`.
- Documents are addressed by **name**, not ID. The agent never sees PageIndex IDs when browsing.
- `StructuredTool.from_function(fn)` wraps each cleanly (schema from the signature, description from
  the docstring, ~570–790 chars). Richer JSON-schema details (the `pages` regex, limit bounds) are not
  carried over; the tools validate their own input.
- Agent run for user A (`gpt-5.6-sol`, Responses API): `browse_documents` → `get_document_structure` →
  `get_page_content(pages="27")` → "60°C (140°F) `<cite doc=\"TDI-110...pdf\" page=\"27\"/>`" — correct.
  `get_citations()` returned `doc_id` for that name; `resolve_citations()` rewrote it to `[[1]](#pageindex-citation-01)`.
- Isolation: B's `browse_documents` lists only B's two documents. B's `get_page_content` / `get_document`
  on A's document name → "Document not found or you do not have access to it". B's `document_context(A_id)`
  and SDK `get_page_content(A_id, ...)` raise `PageIndexAPIError`. B's `get_citations()` on a tag naming
  A's document returns `doc_id: None`.
- Concurrency (same `storage_path`, one process, Windows): 108 read loops (list, tree, page content)
  while a second document was being indexed — 0 errors. Source reading: writes are atomic
  (`os.replace`), but PageIndex's cross-process lock uses `fcntl` and is a **no-op on Windows**; the
  document list and name de-duplication are unprotected there.

**Conclusion:** Q5 — names/signatures above; `StructuredTool.from_function` works. Q6 — yes, tools and
context are bound to the client's `storage_path`; per-user clients isolate. Citations with
`doc_id: None` must be dropped. Q7 — concurrent reads during a write are fine; keep our own per-user
lock around indexing and deletion (PageIndex's lock does not work on Windows or across processes).

## Spike D — figure-page detection (Open Q4)

**Ran:** `spikes/spike_d_detection.py` over all 180 pages (no API cost), plus contact sheets of 109
candidate pages, labelled by eye.

**Result:**
- Raster: icons are 0.1–0.4% of the page; screenshots/photos 4.6–37%. I-Series p38 has images at 4.6%,
  just under the spec's 5%.
- Vector: a drawing count cannot separate illustrations from ruled tables or text pages with warning
  icons. Tables have 90–190 drawings; text pages with icons have up to ~260 curve items; some
  illustrations have fewer (TDI-110 p14: 164 drawings, 246 curves).
- Rule that matched all 28 labelled vector illustrations, with no extra hits:
  take `page.cluster_drawings()`; keep clusters ≥ 1% of the page whose text density is ≤ 5 characters
  per 1% of page area ("graphics, not a table"); a page is a vector-figure page if those clusters add
  up to ≥ 2% of the page.
- Detected figure pages: **60 of 180** (I-Series 30/65, Navio 6/33, TDI-110 7/32, TD Pilot 17/50).

**Conclusion:** Q4 — `MIN_IMAGE_AREA_RATIO = 0.03`; replace `MIN_DRAWINGS_FOR_CHART` with the
cluster rule (`MIN_GRAPHIC_CLUSTER_RATIO = 0.01`, `MAX_CLUSTER_TEXT_DENSITY = 5`,
`MIN_VECTOR_FIGURE_AREA = 0.02`). Re-check on PDFs with real charts when available.

## Extra — vision model for ingestion (Open Q10, part of Q11)

**Ran:** `spikes/extra_vision_models.py`. Described 3 figure pages (170 DPI PNG + page text, Responses
API, `detail="high"`) with `gpt-5.6-luna`, `-terra` and `-sol`.

| Page (truth) | luna | terra | sol |
|---|---|---|---|
| I-Series p30 (3 Great, 5 Good, 1 No data top-centre) | exact | exact | exact |
| TD Pilot p25 (faint "99%" in Track Status box) | missed | missed | **read it** |
| TDI-110 p14 (drawing, callouts 1–13) | good | good | good |

Tokens per page: ~3.2–3.4k input, 200–430 output; 4–8 s per call. Cost per figure page ≈ $0.001
(luna), $0.011 (terra), $0.020 (sol).

**Conclusion:** Q10 — `CHAT_MODEL = gpt-5.6-sol` accepts image input (Spike B). `VISION_MODEL`
default `gpt-5.6-luna`: exact on the key facts, ~20× cheaper than sol. Descriptions are for *finding*
pages; `view_pages` lets sol read exact values. Raise to sol if eval shows missed small text.

## Q11 — time and cost (measured + extrapolated)

- Indexing (measured, 32 pages, luna): 32.6 s, ≈ $0.02. Linear guess for 100 pages: ~100 s, ≈ $0.05.
- Enrichment (extrapolated from Spike D + extra): ~33% of pages are figure pages → ~33 vision calls for
  100 pages: ≈ $0.04 (luna) / ≈ $0.66 (sol); ~6 s each → ~50 s at `VISION_CONCURRENCY = 4`.
- Per query: each 170-DPI page image ≈ 3.0–3.2k input tokens (≈ $0.013 per model call at sol's
  uncached price) for every model call it stays in context — hence the trimming.

## Step 5 checks — PageIndex adapter (2026-09-30)

**Ran:** the opt-in live tests `tests/integration/test_live_indexing.py` and `test_live_pipeline.py`
(two pages cut from TDI-110 each; about a cent in all), with `OPENAI_API_KEY` removed from the process
environment.

**Result:**
- The key reaches PageIndex through `index={"backend": {"api_key": ...}}` (LiteLLM connection params,
  set per indexing call in the indexing thread); indexing works with no key in `os.environ`.
- **Leak found:** `pageindex/utils.py` runs `load_dotenv(find_dotenv(usecwd=True))` on import, and
  LiteLLM runs `load_dotenv()` on import unless `LITELLM_MODE=PRODUCTION`. Either copies the project
  `.env` into `os.environ`. python-dotenv 1.2.3 honours `PYTHON_DOTENV_DISABLED=1` in `load_dotenv()`
  only; `dotenv_values()` (used by pydantic-settings) still reads the file.
- Upload to indexed with every real part: both figure pages (p13 photo, p14 drawing) were detected,
  described by luna, and their `[FIGURE DESCRIPTION pN fig1]` notes came back from `get_page_content`.
- Stored name = our cleaned file name (spaces kept). Deleting a missing `doc_id` raises
  `PageIndexAPIError("... Document not found.")`; the adapter treats it as already gone.
- Isolation re-checked through the adapter: user B's browse is empty, B's `get_page_content` on A's
  name is an error, B's citations for A's name have `doc_id: None`, B's `document_context` raises.

**Conclusion:** keep `PYTHON_DOTENV_DISABLED=1` in `indexing/__init__.py`; an offline test in a fresh
interpreter guards it.

## Not tested (still open)

- Q8 (history vs `document_context()` order): not tested. PageIndex's own chat lanes put the context
  first ("the chat lanes prepend it as the first user message"), matching spec 5.6.
- Agent timeout (`AGENT_TIMEOUT_S`) mechanics; tracing choice (LangSmith vs OpenTelemetry).
- Multi-process deployments (the per-user lock then needs to be cross-process).
- PDFs with real data charts (none in `Data/`).
