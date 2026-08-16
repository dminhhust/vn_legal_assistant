"""Central place for reading environment configuration.

MVP infra footprint is deliberately small: Postgres (profiles,
checklist) + Chroma (legal corpus vectors). No Redis/Celery — this MVP
has no session-memory store or background task queue (see
docs/ARCHITECTURE.md "What this redesign cuts, and why"). Kept as a
plain module-level read rather than a class so it's trivial to import
anywhere without instantiation ceremony.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()  # no-op if no .env file is present (e.g. in containers using real env vars)

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vn_legal_assistant"
)
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
# Single-container deployments (root Dockerfile / HF Spaces) run Chroma
# in embedded PersistentClient mode — file-backed, no separate server.
# Set CHROMA_EMBEDDED_DIR (e.g. /data/chroma) to enable it; unset keeps
# the legacy HTTP-client mode used by docker-compose.
CHROMA_EMBEDDED_DIR = os.getenv("CHROMA_EMBEDDED_DIR") or None
