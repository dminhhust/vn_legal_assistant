"""Deterministic due-date computation (docs/ARCHITECTURE.md §4.3,
§4.6.5: "LLMs are unreliable at exact date math, deterministic code is
not"). This is a minimal stand-in for the full sandboxed Code Execution
Tool described in §4.6.5 — just enough to turn a DeadlineRule into a
concrete date. A general-purpose sandboxed execution environment for
arbitrary agent-generated code (tax calculations, charts, ...) is a
later phase; this module only needs to be correct at one narrow job.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from app.rag.schemas import DeadlineRule

# Safety valve against a misconfigured recurring rule (e.g. period_months=0
# slipping past validation) looping effectively forever.
_MAX_RECURRING_LOOKAHEAD_MONTHS = 120


def compute_due_date(
    rule: DeadlineRule,
    *,
    today: Optional[date] = None,
    event_date: Optional[date] = None,
) -> Optional[date]:
    """Returns the next concrete due date for a rule, or None if the
    rule is missing the fields it needs (e.g. an event_triggered rule
    with no event_date supplied yet — the caller should surface that as
    "pending user input", not a computation error)."""
    today = today or date.today()

    if rule.type == "fixed":
        return _compute_fixed(rule, today)
    if rule.type == "recurring":
        return _compute_recurring(rule, today)
    if rule.type == "event_triggered":
        return _compute_event_triggered(rule, event_date)
    return None


def _compute_fixed(rule: DeadlineRule, today: date) -> Optional[date]:
    if rule.month is None or rule.day is None:
        return None
    candidate = date(today.year, rule.month, rule.day)
    if candidate < today:
        candidate = date(today.year + 1, rule.month, rule.day)
    return candidate


def _compute_recurring(rule: DeadlineRule, today: date) -> Optional[date]:
    if not rule.period_months or rule.day is None:
        return None
    anchor_month = rule.month or 1

    month_offset = 0
    while month_offset <= _MAX_RECURRING_LOOKAHEAD_MONTHS:
        total_month_index = (anchor_month - 1) + month_offset
        year = today.year + total_month_index // 12
        month = total_month_index % 12 + 1
        candidate = date(year, month, rule.day)
        if candidate >= today:
            return candidate
        month_offset += rule.period_months

    return None  # unreachable in practice; guards against a bad period_months value


def _compute_event_triggered(rule: DeadlineRule, event_date: Optional[date]) -> Optional[date]:
    if rule.days_after_event is None or event_date is None:
        return None
    return event_date + timedelta(days=rule.days_after_event)
