"""Unit tests for extraction.py. Uses a fake router with a scripted
`.complete()` — never a real LLM API call."""
from __future__ import annotations

from app.llm.schemas import LLMResponse
from app.rag.extraction import extract_obligations_for_category, extract_obligations_from_chunk
from app.rag.retrieval import RetrievedChunk


class _FakeRouter:
    def __init__(self, obligations: list[dict]):
        self._obligations = obligations
        self.call_count = 0
        self.last_prompt = None

    def complete(self, messages, **kwargs):
        self.call_count += 1
        self.last_prompt = messages[0].content
        return LLMResponse(
            text=None,
            structured_output={"obligations": self._obligations},
            provider="fake",
            model="fake-model",
        )


def _chunk(chunk_id="doc:dieu1", article_number="1", part_count=1, part_index=0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Điều 1. Some obligation\n1. Must do X by March 31.",
        metadata={
            "law_name": "Test Law",
            "article_number": article_number,
            "part_count": part_count,
            "part_index": part_index,
        },
    )


def test_extraction_produces_obligation_item_with_correct_fields():
    fake_router = _FakeRouter(
        [
            {
                "title": "Annual filing",
                "description": "Must file an annual report.",
                "deadline_type": "fixed",
                "deadline_month": 3,
                "deadline_day": 31,
                "period_months": None,
                "days_after_event": None,
                "event_description": None,
                "penalty_summary": "A fine applies.",
            }
        ]
    )

    items = extract_obligations_from_chunk(_chunk(), "tax", router=fake_router)

    assert len(items) == 1
    item = items[0]
    assert item.title == "Annual filing"
    assert item.category == "tax"
    assert item.deadline_rule.type == "fixed"
    assert item.deadline_rule.month == 3
    assert item.deadline_rule.day == 31
    assert item.source_chunk_id == "doc:dieu1"
    assert "Test Law" in item.source_citation
    assert "Điều 1" in item.source_citation


def test_citation_includes_part_info_for_split_articles():
    fake_router = _FakeRouter(
        [{"title": "X", "description": "Y", "deadline_type": "fixed", "deadline_month": 1,
          "deadline_day": 1, "period_months": None, "days_after_event": None,
          "event_description": None, "penalty_summary": "Z"}]
    )
    chunk = _chunk(part_count=2, part_index=1)

    items = extract_obligations_from_chunk(chunk, "tax", router=fake_router)

    assert "part 2/2" in items[0].source_citation


def test_empty_obligations_response_produces_no_items():
    fake_router = _FakeRouter([])
    items = extract_obligations_from_chunk(_chunk(), "tax", router=fake_router)
    assert items == []


def test_extract_for_category_limits_to_top_k_chunks():
    fake_router = _FakeRouter(
        [{"title": "X", "description": "Y", "deadline_type": "fixed", "deadline_month": 1,
          "deadline_day": 1, "period_months": None, "days_after_event": None,
          "event_description": None, "penalty_summary": "Z"}]
    )
    chunks = [_chunk(chunk_id=f"doc:dieu{i}") for i in range(5)]

    extract_obligations_for_category(chunks, "tax", router=fake_router, top_k_chunks=2)

    assert fake_router.call_count == 2  # only the first 2 chunks triggered a call


def test_recurring_deadline_fields_pass_through():
    fake_router = _FakeRouter(
        [{"title": "Quarterly filing", "description": "Y", "deadline_type": "recurring",
          "deadline_month": 1, "deadline_day": 20, "period_months": 3,
          "days_after_event": None, "event_description": None, "penalty_summary": "Z"}]
    )
    items = extract_obligations_from_chunk(_chunk(), "tax", router=fake_router)
    rule = items[0].deadline_rule
    assert rule.type == "recurring"
    assert rule.period_months == 3


def test_event_triggered_deadline_fields_pass_through():
    fake_router = _FakeRouter(
        [{"title": "Registration", "description": "Y", "deadline_type": "event_triggered",
          "deadline_month": None, "deadline_day": None, "period_months": None,
          "days_after_event": 10, "event_description": "signing a labor contract",
          "penalty_summary": "Z"}]
    )
    items = extract_obligations_from_chunk(_chunk(), "tax", router=fake_router)
    rule = items[0].deadline_rule
    assert rule.type == "event_triggered"
    assert rule.days_after_event == 10
    assert rule.event_description == "signing a labor contract"
