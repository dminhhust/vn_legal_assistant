"""Reranking (docs/ARCHITECTURE.md §4.2: "the step that most improves
answer precision in legal RAG, where wrong-but-similar clauses are
common"). Two implementations behind a small Protocol, matching the
provider-abstraction pattern used everywhere else in this codebase
(Model Router, embedding providers):

  - NoOpReranker: keeps the fused RRF order unchanged. Safe default
    when no LLM is configured/available — the pipeline degrades
    gracefully to "RRF order only" rather than failing outright.
  - LLMReranker: asks an LLM (via app.llm.router) to pick and order the
    most relevant candidates for a query, using structured output.
"""
from __future__ import annotations

from typing import Optional, Protocol

from app.llm.schemas import Message
from app.rag.retrieval import RetrievedChunk

_RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "ranked_chunk_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "chunk_ids ordered from most to least relevant to the query.",
        }
    },
    "required": ["ranked_chunk_ids"],
}


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoOpReranker:
    """Identity reranker — keeps whatever order retrieval already
    produced (fused RRF order). See module docstring."""

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return candidates


class LLMReranker:
    """Real reranker using the Model Router. `router` is injectable so
    tests can supply a fake with a scripted `.complete()` — see
    tests/test_rag_reranker.py — never a real API call in automated
    tests."""

    def __init__(self, router=None, task: str = "legal_extraction", top_k: int = 5):
        self._router = router
        self._task = task
        self._top_k = top_k

    def rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []

        router = self._router
        if router is None:
            from app.llm.router import get_router

            router = get_router()

        candidate_desc = "\n".join(f"- {c.chunk_id}: {c.text[:200]}" for c in candidates)
        prompt = (
            f"Query: {query}\n\nCandidates:\n{candidate_desc}\n\n"
            f"Return the chunk_ids of the {self._top_k} most relevant candidates for "
            "the query above, ordered from most to least relevant."
        )
        response = router.complete(
            [Message(role="user", content=prompt)],
            task=self._task,
            response_schema=_RERANK_SCHEMA,
        )
        ranked_ids = (response.structured_output or {}).get("ranked_chunk_ids", [])

        by_id = {c.chunk_id: c for c in candidates}
        reranked = [by_id[cid] for cid in ranked_ids if cid in by_id]

        # Any candidate the LLM didn't mention keeps its original
        # (fused RRF) relative order, appended after the ones it did pick
        # — so a partial/malformed LLM response degrades gracefully
        # instead of silently dropping candidates.
        mentioned = set(ranked_ids)
        reranked.extend(c for c in candidates if c.chunk_id not in mentioned)

        return reranked[: self._top_k] if self._top_k else reranked
