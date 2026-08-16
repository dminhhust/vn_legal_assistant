"""Unit tests for reranker.py. LLMReranker is tested with a fake router
(scripted `.complete()`), never a real API call — same pattern as
tests/test_llm_router.py's fake adapters.
"""
from __future__ import annotations

from app.llm.schemas import LLMResponse
from app.rag.reranker import LLMReranker, NoOpReranker
from app.rag.retrieval import RetrievedChunk


def _chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, text=f"text for {chunk_id}", metadata={})


class _FakeRouter:
    def __init__(self, ranked_ids: list[str]):
        self._ranked_ids = ranked_ids
        self.call_count = 0

    def complete(self, messages, **kwargs):
        self.call_count += 1
        return LLMResponse(
            text=None,
            structured_output={"ranked_chunk_ids": self._ranked_ids},
            provider="fake",
            model="fake-model",
        )


def test_noop_reranker_returns_candidates_unchanged():
    candidates = [_chunk("a"), _chunk("b"), _chunk("c")]
    result = NoOpReranker().rerank("any query", candidates)
    assert result == candidates


def test_noop_reranker_handles_empty_candidates():
    assert NoOpReranker().rerank("any query", []) == []


def test_llm_reranker_reorders_according_to_llm_response():
    candidates = [_chunk("a"), _chunk("b"), _chunk("c")]
    fake_router = _FakeRouter(ranked_ids=["c", "a", "b"])

    result = LLMReranker(router=fake_router, top_k=5).rerank("query", candidates)

    assert [c.chunk_id for c in result] == ["c", "a", "b"]
    assert fake_router.call_count == 1


def test_llm_reranker_respects_top_k():
    candidates = [_chunk("a"), _chunk("b"), _chunk("c")]
    fake_router = _FakeRouter(ranked_ids=["a", "b", "c"])

    result = LLMReranker(router=fake_router, top_k=2).rerank("query", candidates)

    assert len(result) == 2
    assert [c.chunk_id for c in result] == ["a", "b"]


def test_llm_reranker_appends_unmentioned_candidates_after_mentioned_ones():
    candidates = [_chunk("a"), _chunk("b"), _chunk("c")]
    # The fake LLM only mentions "b" — "a" and "c" should still appear,
    # in their original relative order, after "b".
    fake_router = _FakeRouter(ranked_ids=["b"])

    result = LLMReranker(router=fake_router, top_k=5).rerank("query", candidates)

    assert [c.chunk_id for c in result] == ["b", "a", "c"]


def test_llm_reranker_ignores_hallucinated_chunk_ids_not_in_candidates():
    candidates = [_chunk("a"), _chunk("b")]
    fake_router = _FakeRouter(ranked_ids=["a", "made-up-id-that-does-not-exist", "b"])

    result = LLMReranker(router=fake_router, top_k=5).rerank("query", candidates)

    assert [c.chunk_id for c in result] == ["a", "b"]


def test_llm_reranker_returns_empty_for_empty_candidates_without_calling_router():
    fake_router = _FakeRouter(ranked_ids=[])
    result = LLMReranker(router=fake_router).rerank("query", [])
    assert result == []
    assert fake_router.call_count == 0
