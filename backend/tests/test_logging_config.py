"""Unit tests for logging_config.py's JSON formatter."""
from __future__ import annotations

import json
import logging

from app.logging_config import JSONFormatter


def _make_record(msg="hello", level=logging.INFO, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger", level=level, pathname="test.py", lineno=1, msg=msg, args=(), exc_info=None
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_formats_as_valid_json():
    formatted = JSONFormatter().format(_make_record())
    parsed = json.loads(formatted)  # raises if not valid JSON
    assert parsed["message"] == "hello"


def test_includes_expected_core_fields():
    parsed = json.loads(JSONFormatter().format(_make_record(msg="test message", level=logging.WARNING)))
    assert parsed["level"] == "WARNING"
    assert parsed["logger"] == "test.logger"
    assert parsed["message"] == "test message"
    assert "timestamp" in parsed


def test_includes_request_id_when_present():
    parsed = json.loads(JSONFormatter().format(_make_record(request_id="abc-123")))
    assert parsed["request_id"] == "abc-123"


def test_omits_request_id_when_absent():
    parsed = json.loads(JSONFormatter().format(_make_record()))
    assert "request_id" not in parsed


def test_includes_exception_info_when_present():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _make_record(msg="something failed")
        record.exc_info = sys.exc_info()
        parsed = json.loads(JSONFormatter().format(record))

    assert "exception" in parsed
    assert "boom" in parsed["exception"]
