"""Unit tests for deadline.py's deterministic date math."""
from __future__ import annotations

from datetime import date

from app.rag.deadline import compute_due_date
from app.rag.schemas import DeadlineRule


def test_fixed_deadline_this_year_if_not_yet_passed():
    rule = DeadlineRule(type="fixed", month=6, day=15)
    result = compute_due_date(rule, today=date(2026, 1, 1))
    assert result == date(2026, 6, 15)


def test_fixed_deadline_rolls_to_next_year_if_already_passed():
    rule = DeadlineRule(type="fixed", month=3, day=31)
    result = compute_due_date(rule, today=date(2026, 6, 1))
    assert result == date(2027, 3, 31)


def test_fixed_deadline_on_exact_day_counts_as_still_due():
    rule = DeadlineRule(type="fixed", month=3, day=31)
    result = compute_due_date(rule, today=date(2026, 3, 31))
    assert result == date(2026, 3, 31)


def test_fixed_deadline_missing_fields_returns_none():
    rule = DeadlineRule(type="fixed", month=None, day=15)
    assert compute_due_date(rule, today=date(2026, 1, 1)) is None


def test_recurring_quarterly_deadline_finds_next_occurrence():
    # Anchored at January, day 20, every 3 months -> Jan 20 / Apr 20 / Jul 20 / Oct 20.
    rule = DeadlineRule(type="recurring", month=1, day=20, period_months=3)
    result = compute_due_date(rule, today=date(2026, 2, 1))
    assert result == date(2026, 4, 20)


def test_recurring_deadline_on_exact_occurrence_day_counts_as_due():
    rule = DeadlineRule(type="recurring", month=1, day=20, period_months=3)
    result = compute_due_date(rule, today=date(2026, 4, 20))
    assert result == date(2026, 4, 20)


def test_recurring_deadline_wraps_into_next_year():
    rule = DeadlineRule(type="recurring", month=1, day=20, period_months=3)
    result = compute_due_date(rule, today=date(2026, 11, 1))
    assert result == date(2027, 1, 20)


def test_recurring_deadline_defaults_anchor_month_to_january():
    rule = DeadlineRule(type="recurring", month=None, day=15, period_months=6)
    result = compute_due_date(rule, today=date(2026, 1, 1))
    assert result == date(2026, 1, 15)


def test_recurring_deadline_missing_period_returns_none():
    rule = DeadlineRule(type="recurring", month=1, day=15, period_months=None)
    assert compute_due_date(rule, today=date(2026, 1, 1)) is None


def test_event_triggered_deadline_adds_days_to_event_date():
    rule = DeadlineRule(type="event_triggered", days_after_event=10)
    result = compute_due_date(rule, event_date=date(2026, 5, 1))
    assert result == date(2026, 5, 11)


def test_event_triggered_deadline_without_event_date_returns_none():
    rule = DeadlineRule(type="event_triggered", days_after_event=10)
    assert compute_due_date(rule, today=date(2026, 1, 1)) is None


def test_event_triggered_deadline_without_days_after_event_returns_none():
    rule = DeadlineRule(type="event_triggered", days_after_event=None)
    assert compute_due_date(rule, event_date=date(2026, 1, 1)) is None
