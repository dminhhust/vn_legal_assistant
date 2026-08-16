#!/bin/bash
# Single-container entrypoint: database migrate, background corpus
# ingestion (if the vector store is empty), then FastAPI backend +
# Streamlit frontend in the same container.
#
# Env contract (see the root Dockerfile for defaults):
#   DATA_DIR          where SQLite, Chroma data and logs live (persist this)
#   DATABASE_URL      SQLite URL (default sqlite:////data/app.db)
#   CHROMA_EMBEDDED_DIR  embedded Chroma data dir (default /data/chroma)
#   EMBEDDING_PROVIDER   auto|openai|gemini|local|hashing (default local)
#   BACKEND_PORT      uvicorn port (default 8080)
#   FRONTEND_PORT     streamlit port; falls back to $PORT when set
#                     (HF Spaces convention) — default 8501
#   BACKEND_URL       URL the frontend calls (default http://localhost:8080)
set -e

mkdir -p "$DATA_DIR"

FRONTEND_PORT="${PORT:-$FRONTEND_PORT}"

echo "=== [1/5] Initializing database ==="
cd /app/backend
python -m app.db.migrate

echo "=== [2/5] Checking vector store ==="
if python -c "
import sys
from app.ingestion.vector_store import VectorStoreWriter
count = VectorStoreWriter().count()
print(f'vector_store_chunk_count={count}')
sys.exit(0 if count > 100 else 1)
"; then
    echo "Corpus already ingested — skipping background ingestion."
else
    echo "Vector store empty — starting full HF-dataset ingestion in background."
    echo "Progress: tail -f $DATA_DIR/ingestion.log"
    nohup python -m app.ingestion.run_ingestion --hf-dataset --full --batched \
        > "$DATA_DIR/ingestion.log" 2>&1 &
    echo "Background ingestion PID: $!"
fi

echo "=== [3/5] Starting backend (uvicorn :$BACKEND_PORT) ==="
cd /app/backend
nohup uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" \
    > "$DATA_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

echo "=== [4/5] Starting frontend (streamlit :$FRONTEND_PORT) ==="
cd /app/frontend
nohup streamlit run streamlit_app.py \
    --server.address 0.0.0.0 --server.port "$FRONTEND_PORT" \
    > "$DATA_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true' EXIT INT TERM

echo "=== [5/5] Services up (backend pid $BACKEND_PID, frontend pid $FRONTEND_PID) ==="
# Wait on the servers only — the background ingestion is fire-and-forget:
# if it crashes, the app stays up (degraded retrieval) and the crash is
# visible in ingestion.log.
wait $BACKEND_PID $FRONTEND_PID