# Single-container deployment (FastAPI backend + embedded Chroma +
# SQLite + nginx-served React frontend in one image). Deployable as-is
# on any host that runs one Docker container (Hugging Face Spaces,
# Back4app Containers, a single VM, ...). The multi-container stack in
# docker-compose.yml (Postgres + Chroma server + separate services)
# remains the alternative for hosts that support it.
#
# Data persistence: everything stateful lives under /data — SQLite
# (DATABASE_URL), embedded Chroma (CHROMA_EMBEDDED_DIR), the HF model/
# dataset caches (HF_HOME). Persist /data on your host (volume, HF
# Spaces persistent storage, ...) so an already-ingested corpus
# survives restarts.
FROM node:20-slim AS frontend-build

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/app ./backend/app
COPY --from=frontend-build /build/dist /usr/share/nginx/html
COPY frontend/nginx.conf.template /etc/nginx/templates/default.conf.template
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
    BACKEND_PROXY=http://127.0.0.1:8080 \
    PYTHONPATH=/app/backend \
    PYTHONUNBUFFERED=1

EXPOSE 8080 8501

CMD ["/bin/bash", "./entrypoint.sh"]