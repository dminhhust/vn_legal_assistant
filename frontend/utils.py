"""Small shared helper for the Streamlit pages.

Found by actually clicking the real "Generate My Checklist" button
against a real running backend with no LLM key configured: the backend
returns a clean, actionable 503 with a `detail` message (see
backend/app/rag/router.py) — but `requests.Response.raise_for_status()`
only puts the HTTP status line ("503 Server Error: Service
Unavailable...") into the exception, not the JSON body, so the actual
useful message never reached the UI. `friendly_error_message` pulls the
real `detail` field out when present.
"""
from __future__ import annotations

import requests


def friendly_error_message(exc: requests.RequestException) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail")
            if detail:
                return str(detail)
        except (ValueError, AttributeError):
            pass
    return str(exc)
