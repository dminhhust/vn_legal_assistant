---
title: VN Legal Assistant — MVP
emoji: ⚖️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# VN Legal Assistant — MVP

A personal legal assistant for Vietnam: personal info collection, an
auto-generated obligation checklist (manually activated), a RAG legal
chatbot, and a legal-source ingestion pipeline behind the scenes.

- **Backend**: FastAPI — profile service, checklist generator
  (hybrid BM25 + vector retrieval → LLM extraction → deadline
  computation), RAG chat agent (stateless, tools included).
- **Frontend**: React (Vite) SPA served by nginx, which proxies `/api`
  to the backend — no CORS anywhere in the container deployment.
- **Corpus**: real Vietnamese legal documents from the
  `tmquan/vbpl-vn` Hugging Face dataset (~158K documents), ingested
  into embedded Chroma.
- **Deployment shapes**: single container (root `Dockerfile` — React +
  FastAPI + SQLite + embedded Chroma + nginx in one image, this is what
  runs on Hugging Face Spaces) or the multi-container stack in
  `docker-compose.yml` (Postgres + Chroma server + separate services).

## Quick start

Local (needs a running Chroma + Postgres, see `docker-compose.yml`):

```bash
cd backend
python -m app.db.migrate
python -m app.ingestion.run_ingestion          # sample fixture, smoke test
uvicorn app.main:app --host 0.0.0.0 --port 8080
cd ../frontend
npm install
npm run dev                                     # http://localhost:5173
```

## Hugging Face Spaces (single container)

1. Create a Space with **SDK: Docker** and connect this repo (or push
   it to the Space's own git remote).
2. The root `Dockerfile` builds the whole app; `deploy/entrypoint.sh`
   listens on `$PORT` (7860) automatically.
3. Add secrets in Space settings: at least `GOOGLE_API_KEY`
   (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`). Everything else is
   baked in: `EMBEDDING_PROVIDER=local`, SQLite at `/data/app.db`,
   embedded Chroma at `/data/chroma`.
4. First boot: the app serves immediately while the full corpus
   ingests in the background (downloads the e5-small embedding model
   once, then streams + ingests the dataset — roughly 1–2 h on the
   free CPU tier). Progress: `tail -f /data/ingestion.log` via the
   Space's Logs tab.
5. Free-tier note: the container's `/data` resets on a **rebuild** of
   the Space (it survives restarts). HF's paid persistent storage
   keeps `/data` across rebuilds.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:////data/app.db` | SQLite path (single container) |
| `CHROMA_EMBEDDED_DIR` | `/data/chroma` | Embedded Chroma data dir; unset → HTTP Chroma (compose) |
| `EMBEDDING_PROVIDER` | `local` | `auto`/`openai`/`gemini`/`local`/`hashing` |
| `BACKEND_PORT` / `FRONTEND_PORT` | `8080` / `8501` | Ports (frontend falls back to `$PORT` on HF Spaces) |
| `BACKEND_PROXY` | `http://127.0.0.1:8080` | nginx `/api` upstream |
| `DATA_DIR` | `/data` | Where all state + logs live (persist this) |

## Tests

```bash
cd backend
pip install -r requirements.txt
python -m pytest tests          # 258 tests, SQLite in-memory — no services needed
```

## Security note

`backend/.env` is a local template and is gitignored — never commit
real API keys. Rotate any key that was ever exposed.