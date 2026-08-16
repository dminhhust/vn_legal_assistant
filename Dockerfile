# Single-container deployment (backend + frontend + embedded Chroma +
# SQLite in one image). Deployable as-is on any host that runs one
# Docker container (Hugging Face Spaces free CPU basic, Back4app
# Containers, a single VM, ...). The multi-container stack in
# docker-compose.yml (Postgres + Chroma server + separate services)
# remains the alternative for hosts that support it.
#
# Data persistence: everything stateful lives under /data — SQLite
# (DATABASE_URL), embedded Chroma (CHROMA_EMBEDDED_DIR), the HF model/
# dataset caches (HF_HOME). Persist /data on your host (volume, HF
# Spaces persistent storage, ...) so an already-ingested corpus
# survives restarts.
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
COPY frontend/requirements.txt ./frontend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && pip install --no-cache-dir -r frontend/requirements.txt

COPY backend/app ./backend/app
COPY frontend ./frontend
COPY deploy/entrypoint.sh ./entrypoint.sh
RUN chmod +x ./entrypoint.sh

# --- single-container defaults (all overridable at run time) ---
ENV DATA_DIR=/data \
    DATABASE_URL=sqlite:////data/app.db \
    CHROMA_EMBEDDED_DIR=/data/chroma \
    HF_HOME=/data/hf \
    EMBEDDING_PROVIDER=local \
    BACKEND_PORT=8080 \
    FRONTEND_PORT=8501 \
    BACKEND_URL=http://localhost:8080 \
    PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1

EXPOSE 8080 8501

CMD ["/bin/bash", "./entrypoint.sh"]