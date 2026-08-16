"""Structured logging (Phase 9 hardening, docs/IMPLEMENTATION_PLAN.md
Phase 9: "structured logging + basic tracing"). Plain stdlib logging
with a JSON formatter — no external logging service dependency, since
this is about giving log lines a consistent, parseable shape (so a
real deployment can ship them to whatever log aggregator it uses), not
about standing up observability infrastructure this sandbox can't
verify anyway.
"""
from __future__ import annotations

import json
import logging
import sys


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: int = logging.INFO) -> None:
    """Replaces the root logger's handlers with a single JSON-formatted
    stdout handler. Idempotent — safe to call more than once (e.g. once
    at import time, once again in a test) without accumulating
    duplicate handlers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
