#!/usr/bin/env bash
# One-shot deployment for Google Colab (free tier). Serves the entire
# app (React + FastAPI + SQLite + embedded Chroma + nginx) behind a
# Cloudflare quick tunnel — no account, no card, no repo changes.
#
# Run from a Colab cell:
#   !bash deploy/colab_setup.sh
# Then copy the trycloudflare.com URL printed at the end.
#
# Persistence: SQLite/Chroma/model cache live on Google Drive
# (MyDrive/vn_legal_data), so the 1-2h corpus ingest runs only once.
# Caveats: free Colab disconnects after ~90 min idle / ~12 h uptime;
# the tunnel URL changes every session.
set -euo pipefail

ROOT=/content/vn_legal_assistant
DATA_DIR=/content/drive/MyDrive/vn_legal_data
PORT="${PORT:-8501}"

echo "==> system deps (nginx)"
apt-get update -qq >/dev/null
apt-get install -y -qq nginx gettext-base >/dev/null
command -v node >/dev/null || apt-get install -y -qq nodejs npm >/dev/null

echo "==> code"
if [ ! -d "$ROOT/.git" ]; then
  git clone --quiet https://github.com/dminhhust/vn_legal_assistant "$ROOT"
fi
cd "$ROOT"
git pull --quiet --ff-only || true

echo "==> python deps"
pip install -q -r backend/requirements.txt

echo "==> frontend build"
cd frontend
npm install --silent >/dev/null
npm run build >/dev/null
cd ..

echo "==> env"
mkdir -p "$DATA_DIR"
export DATA_DIR
export DATABASE_URL="sqlite:///$DATA_DIR/app.db"
export CHROMA_EMBEDDED_DIR="$DATA_DIR/chroma"
export EMBEDDING_PROVIDER=local
export HF_HOME="$DATA_DIR/hf"          # keep the e5-small model on Drive
export BACKEND_PROXY=http://127.0.0.1:8080
export FRONTEND_PORT="$PORT"

echo "==> db + corpus (skipped once ingested)"
cd backend
python -m app.db.migrate
if [ ! -d "$DATA_DIR/chroma" ] || [ -z "$(ls -A "$DATA_DIR/chroma" 2>/dev/null)" ]; then
  python -m app.ingestion.run_ingestion --hf-dataset --full --batched
else
  echo "    corpus already present - skipping ingest"
fi

echo "==> nginx (serve dist + proxy /api)"
rm -rf /usr/share/nginx/html/*
cp -r ../frontend/dist/* /usr/share/nginx/html/
rm -f /etc/nginx/sites-enabled/default
envsubst '${FRONTEND_PORT} ${BACKEND_PROXY}' \
  < ../frontend/nginx.conf.template \
  > /etc/nginx/sites-enabled/default
nginx -t >/dev/null
service nginx start

echo "==> backend"
nohup python -m uvicorn app.main:app --host 127.0.0.1 --port 8080 \
  > /tmp/backend.log 2>&1 &
sleep 8
curl -fsS http://127.0.0.1:8080/health || { tail -n 40 /tmp/backend.log; exit 1; }

echo "==> tunnel (Ctrl+C ends the session)"
pip install -q pycloudflared
pycloudflared --url "http://127.0.0.1:$PORT"