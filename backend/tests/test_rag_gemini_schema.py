"""Unit tests for the Gemini adapter's response-schema normalization.

These tests never call a real LLM API and never require the google-genai
SDK — they exercise `GeminiAdapter._sanitize_schema_for_gemini`, the
pure function that translates this codebase's shared JSON-Schema style
(lowercase types, multi-type arrays like ["integer", "null"]) into the
uppercase single-enum + `nullable` dialect the Gemini API requires.

Regression context: checklist generation returned a clean-but-useless
503 ("all 5 key(s) exhausted") against the live Gemini API because
OBLIGATION_EXTRACTION_SCHEMA's `["integer", "null"]` type lists made the
API reject every request before any model call happened. This test
locks the normalization in so that failure mode can't return.
"""
from __future__ import annotations

from app.llm.adapters.gemini_adapter import GeminiAdapter

_SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "deadline_month": {"type": ["integer", "null"]},
                    "event_description": {"type": ["string", "null"]},
                    "deadline_type": {
                        "type": "string",
                        "enum": ["fixed", "recurring", "event_triggered"],
                    },
                },
            },
        }
    },
}


def test_sanitizes_multi_type_arrays_into_nullable_flags() -> None:
    out = GeminiAdapter._sanitize_schema_for_gemini(_SAMPLE_SCHEMA)

    deadline_month = out["properties"]["obligations"]["items"]["properties"]["deadline_month"]
    assert deadline_month == {"type": "INTEGER", "nullable": True}

    event_description = out["properties"]["obligations"]["items"]["properties"][
        "event_description"
    ]
    assert event_description == {"type": "STRING", "nullable": True}


def test_uppercases_plain_types_and_keeps_other_keys() -> None:
    out = GeminiAdapter._sanitize_schema_for_gemini(_SAMPLE_SCHEMA)

    assert out["type"] == "OBJECT"
    title = out["properties"]["obligations"]["items"]["properties"]["title"]
    assert title == {"type": "STRING"}

    deadline_type = out["properties"]["obligations"]["items"]["properties"]["deadline_type"]
    assert deadline_type == {
        "type": "STRING",
        "enum": ["fixed", "recurring", "event_triggered"],
    }


def test_handles_pure_null_type_and_multi_non_null_unions() -> None:
    out = GeminiAdapter._sanitize_schema_for_gemini(
        {
            "type": "object",
            "properties": {
                "only_null": {"type": ["null"]},
                "union": {"type": ["integer", "string"]},
            },
        }
    )

    props = out["properties"]
    assert props["only_null"] == {"type": "NULL"}
    # Gemini can't express multi-type unions; degrade permissively to
    # STRING rather than fail the whole call.
    assert props["union"] == {"type": "STRING"}


def test_tool_parameters_are_sanitized_too() -> None:
    from app.chat.schemas import ALL_CHAT_TOOLS

    for tool in ALL_CHAT_TOOLS:
        out = GeminiAdapter._to_gemini_tool(tool)
        declaration = out["function_declarations"][0]
        assert declaration["name"] == tool.name
        assert declaration["parameters"]["type"] == "OBJECT"

        days_ahead = (
            declaration["parameters"].get("properties", {}).get("days_ahead")
        )
        if days_ahead is not None:
            assert days_ahead["type"] == "INTEGER"
            assert days_ahead["nullable"] is True
