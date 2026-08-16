"""Regression test for a real bug found by running this app against a
live Postgres instance rather than only the (SQLite, in-process) test
suite: `init_db()` reported success while creating ZERO tables,
because `Base.metadata.create_all()` only knows about a table once the
module defining its model class has been imported somewhere, and
nothing had imported app.db.models yet.

This is deliberately run in an ISOLATED SUBPROCESS rather than as a
normal in-process pytest test. In-process, this bug is invisible: by
the time any test in this suite calls create_all(), some OTHER test
module earlier in the same pytest session has almost certainly already
done `from app.db.models import ...` (e.g. test_checklist_api.py does),
which populates the process-wide `Base.metadata` regardless of what
init_db() itself does. A normal test would pass whether or not the fix
in app/db/session.py's init_db() (importing app.db.models internally)
is actually present — exactly why this bug survived 161 passing tests
and was only caught by running a real, freshly-started process against
a real database.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

EXPECTED_TABLES = {"users", "profiles", "profile_history", "trait_tags", "obligation_checklist_items"}


def test_init_db_creates_all_tables_in_a_fresh_process():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "fresh.db"
        script = (
            "from app.db.session import init_db, engine\n"
            "init_db()\n"
            "from sqlalchemy import inspect\n"
            "tables = inspect(engine).get_table_names()\n"
            "print('TABLES:' + ','.join(sorted(tables)))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            env={"DATABASE_URL": f"sqlite:///{db_path}", "PATH": "/usr/bin:/bin"},
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"

        table_line = next(line for line in result.stdout.splitlines() if line.startswith("TABLES:"))
        actual_tables = set(table_line.removeprefix("TABLES:").split(","))

        assert EXPECTED_TABLES.issubset(actual_tables), (
            f"init_db() did not create all expected tables in a fresh process. "
            f"Expected at least {EXPECTED_TABLES}, got {actual_tables}. "
            "This is the exact bug found running against a real Postgres instance: "
            "init_db() reported success but created zero tables because "
            "app.db.models was never imported first."
        )


class TestWaitForDatabase:
    """Regression tests for a second real bug found by actually running
    `docker compose up`: on a first run, Postgres takes a few seconds
    to finish its own post-bootstrap init after the container starts,
    and `python -m app.db.migrate` raced it — crashing with a raw
    `psycopg2.OperationalError` traceback instead of waiting. Fixed
    with a bounded retry loop in app/db/migrate.py; docker-compose.yml
    also now gates the backend's startup on Postgres's own healthcheck,
    but that's an orchestration-level mitigation — this code-level
    retry is what makes `python -m app.db.migrate` robust even when run
    outside docker-compose (or with a slow-starting managed DB).
    """

    def test_retries_until_the_database_becomes_available(self, monkeypatch):
        import app.db.migrate as migrate_module
        from sqlalchemy.exc import OperationalError

        monkeypatch.setattr(migrate_module, "MAX_WAIT_SECONDS", 5)
        monkeypatch.setattr(migrate_module, "RETRY_INTERVAL_SECONDS", 0.01)

        call_count = {"n": 0}

        class _FakeConnectionContext:
            def __enter__(self_inner):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    raise OperationalError("connect", {}, Exception("connection refused"))
                return object()

            def __exit__(self_inner, *args):
                return False

        monkeypatch.setattr(migrate_module.engine, "connect", lambda: _FakeConnectionContext())

        migrate_module._wait_for_database()  # should NOT raise/exit — succeeds on the 3rd attempt

        assert call_count["n"] == 3

    def test_gives_up_with_a_clean_exit_if_the_database_never_comes_up(self, monkeypatch):
        """Confirms the retry loop is bounded (this is what actually ran
        for real in the earlier manual verification: exit code 1 after
        the timeout, with a clear message, not a hang and not a raw
        traceback)."""
        import app.db.migrate as migrate_module
        from sqlalchemy.exc import OperationalError

        monkeypatch.setattr(migrate_module, "MAX_WAIT_SECONDS", 0.05)
        monkeypatch.setattr(migrate_module, "RETRY_INTERVAL_SECONDS", 0.01)

        class _AlwaysFailsConnectionContext:
            def __enter__(self_inner):
                raise OperationalError("connect", {}, Exception("connection refused"))

            def __exit__(self_inner, *args):
                return False

        monkeypatch.setattr(migrate_module.engine, "connect", lambda: _AlwaysFailsConnectionContext())

        with pytest.raises(SystemExit) as exc_info:
            migrate_module._wait_for_database()

        assert exc_info.value.code == 1
