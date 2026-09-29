# Vectorless RAG Service

A multi-user service that ingests PDFs, writes figure descriptions into them as invisible text, indexes
them with PageIndex (local mode, no vector DB), and answers questions with a LangChain agent that can
look at page images itself. Answers carry page-level citations. Each user sees only their own documents.

- Spec (source of truth): `docs/vectorless-rag-spec.md` (v0.4). Read it before designing anything.
- Verified facts behind the spec: `docs/spike-findings.md`.
- If guidance conflicts: locked decisions in the spec > the rest of the spec > the
  python-clean-architecture plugin. Flag conflicts to the user instead of choosing silently.

## Locked decisions (spec Sections 3 and 5.7; do not change)

- Index mode: PageIndex **local**, on our OpenAI key.
- Model provider: OpenAI.
- Agent framework: LangChain.
- Index-side images: a vision model describes figures; descriptions go into the PDF as an invisible text layer before indexing.
- Retrieval-side images: `view_pages` returns original page images to the **main agent**, which reads them itself. No separate vision call.
- Chat model must be vision-capable.
- PageIndex → LangChain: wrap `client.agent_tools()` as LangChain tools.
- System prompt: `agent_instructions()` + `citation_prompt()` + our figure guidance.
- PageIndex tools are read-only (`include_management=False`).
- User isolation: one PageIndex `storage_path` per user.
- Ingestion runs in a background worker.
- Our own database is the source of truth.

## Stack

Python 3.13 · pageindex 0.2.20 · langchain 1.4.3 / langchain-core 1.6.5 / langchain-openai 1.6.6 ·
openai 2.54.0 (3.x conflicts with pageindex) · pymupdf 1.28.2 · FastAPI · SQLAlchemy 2 (SQLite dev,
Postgres prod) · pydantic / pydantic-settings · pytest · pyright. Models: `gpt-5.6-sol` (chat),
`gpt-5.6-luna` (indexing and figure descriptions). Full table: spec Section 17.

## Commands (Windows; Git Bash paths)

```bash
# Service
cp .env.example .env                      # then set OPENAI_API_KEY; every setting is listed there
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/pyright                     # src + tests; must report 0 errors
.venv/Scripts/python -m pytest
.venv/Scripts/python -m uvicorn vectorless_rag.api.main:app --reload   # once api/ exists

# Spikes (throwaway; own venv and pyright config)
python -m venv spikes/.venv && spikes/.venv/Scripts/python -m pip install -r spikes/requirements.txt
.venv/Scripts/pyright -p spikes
```

Restart Claude Code after changing `pyrightconfig.json` so the pyright LSP reloads.

## Layout

```
docs/            spec and spike findings
Data/            sample PDFs: read-only test input, never write here
spikes/          throwaway experiments (own venv); outputs in spikes/out/
src/vectorless_rag/   (spec Section 18; filled in step by step)
  api/ worker/        entry points and composition roots
  operations/         use cases + ports.py (Protocols) + errors.py
  db/ pdf/ vision/ indexing/ agent/ storage/   adapters behind the ports
  models/ config.py
tests/  fakes/ (in-memory ports) unit/ integration/
var/             runtime data (DATA_ROOT): uploads, enriched PDFs, PageIndex storage, SQLite
```

## Conventions

- Keys only from the environment / `.env`. Never hard-code, print or log them. Only the composition
  roots call `load_settings()`; adapters receive values (including the API key) as arguments.
- `operations/` imports only `ports.py`, `models/` and its own modules — never an adapter or SDK.
  Each new adapter must pass the same contract tests as its fake (see `tests/unit/test_repository_contracts.py`).
- No vector DB, no embeddings, no chunking.
- PageIndex agent tools stay read-only.
- One PageIndex client and `storage_path` per user (`var/users/{user_key}/pageindex`); never share
  a client across users. Directory names use internal keys, never user-supplied strings.
- `view_pages(doc_name, pages)` renders the **original** PDF, never the enriched copy, and
  checks ownership through the registry.
- The agent uses the OpenAI **Responses API** (`ChatOpenAI(..., use_responses_api=True)`). Chat
  Completions silently drops images in tool results.
- Page images go to the agent as `image_url` blocks carrying `detail`.
- The enriched PDF keeps the original file name: PageIndex shows the file's base name to the agent
  and in citations.
- Guard indexing entry points with `if __name__ == "__main__":` (PageIndex spawns processes on Windows).
- Interfaces: **Protocol** around third-party SDKs (PageIndex, OpenAI, LangChain) and anything we
  replace with a fake in tests. **ABC** only when implementations share real code or state, or we
  want runtime enforcement. Keep inheritance shallow. Use classes where there is state or
  dependencies; plain functions for stateless steps.
- Python code follows the python-clean-architecture plugin (skill `clean-architecture`), adapted as
  in spec Section 18. Our Protocol-vs-ABC rule above overrides its Protocol-first default.
- pyright must pass with 0 errors.
- Run `/review-architecture` before a feature is considered done.
