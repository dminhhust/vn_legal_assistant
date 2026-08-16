"""Thin CLI wrapper so `python -m app.db.migrate` also works, matching
common convention. The real logic lives in app.db.session.init_db —
this file exists only for a friendlier command name.

Retries the initial connection for a bit before calling init_db().
docker-compose.yml also gates the backend's startup on Postgres's own
healthcheck now, but that's an orchestration-level fix — this retry
loop is a code-level one, so `python -m app.db.migrate` is still
robust if run directly against a database that's still starting (a
different orchestrator without health-gated dependencies, a slow
first-boot on a managed DB, etc.). Found to matter by actually running
`docker compose up`: a fresh `pgdata` volume means Postgres runs a
post-bootstrap init step that takes a few seconds, and this script
raced it and crashed with "Connection refused" before either fix was
in place — see docs/ARCHITECTURE.md §9.
"""
from __future__ import annotations

import sys
import time

from sqlalchemy.exc import OperationalError

from app.db.session import engine, init_db

MAX_WAIT_SECONDS = 30
RETRY_INTERVAL_SECONDS = 2


def _wait_for_database() -> None:
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with engine.connect():
                return
        except OperationalError as exc:
            last_error = exc
            time.sleep(RETRY_INTERVAL_SECONDS)
    print(
        f"Could not connect to the database after {MAX_WAIT_SECONDS}s: {last_error}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    _wait_for_database()
    init_db()
    print("Database tables created (or already existed).")
