"""Data structures for the Legal RAG + Checklist Generator (Phase 3).
See docs/ARCHITECTURE.md §4.3.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

DeadlineType = Literal["fixed", "recurring", "event_triggered"]


@dataclass
class DeadlineRule:
    """Vietnamese obligations aren't all fixed-date, so this supports
    the three shapes from docs/ARCHITECTURE.md §4.3:
      - fixed: same month/day every year (e.g. PIT finalization Mar 31)
      - recurring: due on `day` of the month, every `period_months`,
        anchored at `month` (e.g. quarterly VAT filing)
      - event_triggered: `days_after_event` days after a triggering
        event (e.g. "within 10 days of signing a labor contract")
    """

    type: DeadlineType
    month: Optional[int] = None  # "fixed": the due month; "recurring": the anchor month
    day: Optional[int] = None  # day of month the obligation is due
    period_months: Optional[int] = None  # "recurring": repeat every N months
    days_after_event: Optional[int] = None  # "event_triggered": N days after the trigger
    event_description: Optional[str] = None  # "event_triggered": human description of the trigger


@dataclass
class ObligationItem:
    title: str
    category: str
    description: str
    deadline_rule: DeadlineRule
    penalty_summary: str
    source_citation: str
    source_chunk_id: str


# JSON schema used to force structured LLM output during extraction
# (extraction.py) — see docs/ARCHITECTURE.md §4.3: "structured output
# ... never free text."
OBLIGATION_EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "obligations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "deadline_type": {
                        "type": "string",
                        "enum": ["fixed", "recurring", "event_triggered"],
                    },
                    "deadline_month": {"type": ["integer", "null"]},
                    "deadline_day": {"type": ["integer", "null"]},
                    "period_months": {"type": ["integer", "null"]},
                    "days_after_event": {"type": ["integer", "null"]},
                    "event_description": {"type": ["string", "null"]},
                    "penalty_summary": {"type": "string"},
                },
                "required": ["title", "description", "deadline_type", "penalty_summary"],
            },
        }
    },
    "required": ["obligations"],
}
