# VN Legal Assistant — MVP

A focused, working MVP: personal info collection → an auto-generated
legal obligation checklist you trigger manually → a RAG chatbot
grounded in a real ingested legal corpus → a real crawler for
Vietnamese legal sources feeding that corpus.

This is a redesign of a much larger uploaded prototype, scoped down to
exactly these four features. See `docs/ARCHITECTURE.md` for the full
design and an explicit list of what was cut and why, and
`docs/ORIGINAL_PROTOTYPE_NOTES.md` for notes on the original project
(including a security issue found in it — read that before reusing
anything from the original zip).

## Quick start

```bash
cp backend/.env.example backend/.env
# edit backend/.env — set at least one of ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY

docker compose up --build
```

- Backend: http://localhost:8080 (docs at `/docs`)
- Frontend: http://localhost:8501

The backend container runs `python -m app.db.migrate` on startup to
create tables. The vector store starts empty — see "Loading legal
data" below before generating a checklist or chatting.

## Loading legal data

Three options, all via `backend/app/ingestion/run_ingestion.py`:

```bash
# Inside the backend container (or a local venv with the same DATABASE_URL/CHROMA_HOST):

# Option A — synthetic fixture, for a quick pipeline smoke test:
python -m app.ingestion.run_ingestion

# Option B — real documents, live from the VBPL (Ministry of Justice) gateway:
python -m app.ingestion.run_ingestion --crawl --max-documents 10

# Option C — real documents, streamed from the tmquan/vbpl-vn dataset on Hugging Face:
python -m app.ingestion.run_ingestion --hf-dataset --max-documents 10
```

**Option B is currently broken** — confirmed by actually running it,
not just documented as untested: the live gateway rejects every
request with `400 Bad Request`. It's gated behind a Bearer token the
official vbpl.vn single-page app obtains by solving Google's invisible
reCAPTCHA v2 in a real browser session; a plain HTTP client (which is
all this crawler uses) can't get past that. See
`app/ingestion/crawler.py`'s module docstring for the full root-cause
writeup. This isn't a schema-drift problem — the endpoint paths and
JSON field names the crawler assumes are confirmed correct.

**Option C is the one that currently works.** It sidesteps the auth
wall by streaming from a dataset someone else already crawled properly
(with the necessary reCAPTCHA/token handling) from that same official
source — 158,822 real Vietnamese legal documents, CC-BY-4.0, captured
2026-05-23. It needs network access to `huggingface.co` and the
`datasets` package (in `requirements.txt`). See
`app/ingestion/hf_dataset_loader.py`'s module docstring for the full
schema mapping (confirmed against the dataset's own card and its
build pipeline's source on GitHub, not guessed) and its limitations
(it's a point-in-time mirror, not live-current). Notably: it excludes
12 of the dataset's `doc_type` values by default (translations,
correspondence, notices, international agreements/protocols/MOUs, and
undifferentiated "other/related" catch-alls) since none of them is a
binding legal instrument under the same law the dataset's own
`doc_type` taxonomy follows — see `DEFAULT_EXCLUDED_DOC_TYPES` for the
full list and why each entry is there, and override via
`HfVbplDatasetLoader(excluded_doc_types=...)` if you want any of them
anyway.
As with any ingested source, spot-check the first few ingested
documents by hand before trusting the output.

### Note on retrieval: two pipelines exist, only one is wired up

`app/rag/retrieval.py` (`HybridRetriever`) is what the live app
actually uses — BM25 + vector search, fused via RRF, over the
chunk-level data any of the three ingestion options above write into
Chroma. This is what `app/chat/tools.py` and
`app/rag/checklist_service.py` call.

`app/rag/obligation_retrieval.py` (`ObligationRetriever`) is a
separate, independently-tested, document-level pipeline purpose-built
around `tmquan/vbpl-vn`'s own schema — instrument-hierarchy ranking
(`hierarchy.py`), jurisdiction facets (`jurisdiction.py`), a
consolidation/citation-graph walk (`consolidation.py`), and coverage/
gap scoring (`coverage.py`), all keyed off the dataset's real
`doc_type` slugs and `structure_json` shape. It is **not called from
any router or endpoint** — wiring it in (replacing `HybridRetriever`,
running alongside it, or something else) is a real product decision,
not done here. See `obligation_retrieval.py`'s own module docstring
for the full story of why these are two separate files.

## Local development without Docker

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.db.migrate          # requires DATABASE_URL reachable, e.g. via `docker compose up postgres chroma`
uvicorn app.main:app --reload --port 8080
```

```bash
cd frontend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
BACKEND_URL=http://localhost:8080 streamlit run streamlit_app.py
```

## Running tests

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q
```

All 168 tests run against SQLite + in-memory Chroma + fake LLM
routers — no real database, vector server, or API key required.
This MVP has also been verified against a real, running stack
(real PostgreSQL, real Chroma, real backend, real frontend clicks, and
a real `docker compose up` run) — see `docs/ARCHITECTURE.md` §9 for
the six real bugs that surfaced only from actually running it, and how
they were fixed.

### Verified against the real stack, not just this test suite

This MVP was also checked by actually standing up a real PostgreSQL
server, a real Chroma server, and the real FastAPI app, and driving it
with `curl` — not just `TestClient`. That pass caught and fixed four
real bugs the test suite alone missed (a stale module reference, a
migration that silently created zero tables, a demo checklist that
was silently empty, and two endpoints returning bare 500s instead of a
clean error when no LLM key is set). See `docs/ARCHITECTURE.md` §9 for
the full account and what still needs verifying against real
credentials.

## Project layout

```
backend/app/
  profile/     — personal info collection (onboarding, traits, history)
  ingestion/   — parser, chunker, embeddings, vector store, pipeline,
                 and crawler.py (the legal-source crawler)
  rag/         — category queries, hybrid retrieval, reranking,
                 LLM extraction, deadline math, checklist service + router
                 (POST /checklist/{user_id}/generate is the manual
                 activation endpoint), plus obligation_retrieval.py —
                 a separate, not-yet-wired-in retrieval pipeline (see
                 README "Note on retrieval" above)
  chat/        — the RAG chatbot agent, tools, and router
  llm/         — multi-provider LLM router (Claude / OpenAI / Gemini)
  db/          — SQLAlchemy models + session setup
frontend/
  streamlit_app.py, pages/1_Onboarding.py, pages/2_Dashboard.py, pages/3_Chat.py
docs/
  ARCHITECTURE.md, ORIGINAL_PROTOTYPE_NOTES.md
```
