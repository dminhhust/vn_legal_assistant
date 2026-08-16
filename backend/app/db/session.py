"""Database engine/session setup.

Synchronous SQLAlchemy engine + sessionmaker. Works against either the
real Postgres URL from app.config (used by docker-compose / production)
or an in-memory SQLite URL (used by tests) — see
backend/tests/test_profile_api.py for how tests override `get_db` to
swap in a throwaway SQLite engine per test.
"""
from __future__ import annotations

from collections.abc import Generator
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/vn_legal_assistant"
)


class Base(DeclarativeBase):
    pass


# `check_same_thread` is only meaningful for SQLite; Postgres ignores it
# being absent, so this keeps the same engine-construction code correct
# for both the real DB and the SQLite URL used in tests.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't exist.

    Deliberately NOT called automatically on app startup — auto-DDL on
    every boot is unsafe once there's real data. Call this explicitly:
        python -m app.db.migrate
    Replace with real Alembic migrations before this ever touches a
    production database with data worth preserving.

    The import below is NOT unused — `Base.metadata` only knows about a
    table if the module defining that table's model class has been
    imported somewhere first (that's what registers it with
    DeclarativeBase). This module (session.py) never imports
    app.db.models itself elsewhere, so without this line, calling
    init_db() before anything else has imported app.db.models silently
    creates ZERO tables and reports success — this is a real bug this
    project's own end-to-end run against a live Postgres instance
    caught, not a hypothetical.
    """
    import app.db.models  # noqa: F401 — see docstring above

    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_db()
    print("Database tables created (or already existed).")
