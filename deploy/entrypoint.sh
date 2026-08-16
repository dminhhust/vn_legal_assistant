#!/bin/bash
# Single-container entrypoint: database migrate, background corpus
# ingestion (if the vector store is empty), then FastAPI backend +
# nginx (React bundle + /api proxy) in the same container.
#
# Env contract (see the root Dockerfile for defaults):
#   DATA_DIR          where SQLite, Chroma data and logs live (persist this)
#   DATABASE_URL      SQLite URL (default sqlite:////data/app.db)
#   CHROMA_EMBEDDED_DIR  embedded Chroma data dir (default /data/chroma)
#   EMBEDDING_PROVIDER   auto|openai|gemini|local|hashing (default local)
#   BACKEND_PORT      uvicorn port (default 8080)
#   FRONTEND_PORT     nginx port; falls back to $PORT when set
#                     (HF Spaces convention) — default 8501
#   BACKEND_PROXY     backend URL nginx forwards /api to (default
#                     http://127.0.0.1:8080 — same container)
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

echo "=== [4/5] Configuring nginx (frontend :$FRONTEND_PORT, /api -> $BACKEND_PROXY) ==="
envsubst '${FRONTEND_PORT} ${BACKEND_PROXY}' \
    < /etc/nginx/templates/default.conf.template \
    > /etc/nginx/conf.d/default.conf
nginx -g 'daemon off;' > "$DATA_DIR/nginx.log" 2>&1 &
NGINX_PID=$!

trap 'kill $BACKEND_PID $NGINX_PID 2>/dev/null || true' EXIT INT TERM

echo "=== [5/5] Services up (backend pid $BACKEND_PID, nginx pid $NGINX_PID) ==="
# Wait on the servers only — the background ingestion is fire-and-forget:
# if it crashes, the app stays up (degraded retrieval) and the crash is
# visible in ingestion.log.
wait $BACKEND_PID $NGINX_PID