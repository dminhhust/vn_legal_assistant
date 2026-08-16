"""Orchestrates Phase 3 end to end: user profile/traits -> category
queries -> hybrid retrieval -> (optional) reranking -> LLM extraction
-> deterministic due-date computation -> persisted checklist items.
See docs/ARCHITECTURE.md §4.2-§4.3 and docs/IMPLEMENTATION_PLAN.md
Phase 3's Definition of Done.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import ObligationChecklistItem, User
from app.ingestion.embeddings import EmbeddingProvider, get_default_embedding_provider
from app.ingestion.vector_store import VectorStoreWriter
from app.rag.deadline import compute_due_date
from app.rag.extraction import extract_obligations_for_category
from app.rag.query_builder import build_category_queries, infer_entity_type
from app.rag.reranker import NoOpReranker, Reranker
from app.rag.retrieval import HybridRetriever

logger = logging.getLogger(__name__)


class UserNotFoundError(Exception):
    pass


# There are at most 7 applicable categories; cap the worker pool there.
# Verified empirically (Iteration 10): 7 concurrent Gemini LLM calls
# complete in ~3 s with zero 429s, so the ~97 s sequential checklist was
# serial code, not a rate limit.
_MAX_PARALLEL_CATEGORIES = 7


def _extract_for_category(
    cq,
    entity_type: str,
    province: Optional[str],
    retriever: HybridRetriever,
    reranker: Reranker,
    top_k_chunks: int,
    llm_router,
) -> list:
    """One worker's full per-category pass: retrieve -> rerank -> extract.
    All shared collaborators are thread-safe: the router's GeminiAdapter
    rotates keys under its own lock, and the embedder is lock-guarded
    (see GeminiEmbeddingProvider)."""
    candidates = retriever.retrieve(
        cq, user_entity_type=entity_type, user_province=province
    )
    ranked = reranker.rerank(cq.query_text, candidates)
    return extract_obligations_for_category(
        ranked, cq.category, router=llm_router, top_k_chunks=top_k_chunks
    )


def generate_checklist_for_user(
    db: Session,
    user_id: str,
    *,
    vector_store: Optional[VectorStoreWriter] = None,
    embedder: Optional[EmbeddingProvider] = None,
    retriever: Optional[HybridRetriever] = None,
    reranker: Optional[Reranker] = None,
    llm_router=None,
    top_k_chunks_per_category: int = 3,
    today: Optional[date] = None,
) -> list[ObligationChecklistItem]:
    """Runs the full pipeline for one user and persists the result.

    Wholesale regeneration: any previous checklist rows for this user
    are deleted before the new ones are written. This is a known
    prototype-simple gap — it doesn't preserve user-set status
    (done/dismissed) across a regeneration; a real implementation would
    diff against the previous set instead. Flagged rather than silently
    shipped, matching this codebase's existing pattern (see e.g.
    app/ingestion/vector_store.py's content-hash diffing, which this
    intentionally does NOT replicate at the checklist level yet).
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or user.profile is None:
        raise UserNotFoundError(user_id)

    trait_tags = [t.tag for t in user.traits]
    entity_type = infer_entity_type(trait_tags)
    province = user.profile.province

    embedder = embedder or get_default_embedding_provider()
    store = vector_store or VectorStoreWriter()
    active_retriever = retriever or HybridRetriever(store, embedder)
    active_reranker = reranker or NoOpReranker()

    category_queries = build_category_queries(trait_tags)
    logger.info(
        "Generating checklist for user %s: %d applicable categories (%s)",
        user_id,
        len(category_queries),
        [cq.category for cq in category_queries],
    )

    obligation_items = []
    workers = min(_MAX_PARALLEL_CATEGORIES, len(category_queries)) or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # pool.map preserves category_queries order, so saved items stay
        # in deterministic category order (as the pre-parallel loop did).
        for items in pool.map(
            lambda args: _extract_for_category(*args),
            [
                (
                    cq,
                    entity_type,
                    province,
                    active_retriever,
                    active_reranker,
                    top_k_chunks_per_category,
                    llm_router,
                )
                for cq in category_queries
            ],
        ):
            obligation_items.extend(items)

    db.query(ObligationChecklistItem).filter(ObligationChecklistItem.user_id == user_id).delete()

    saved: list[ObligationChecklistItem] = []
    for item in obligation_items:
        due = compute_due_date(item.deadline_rule, today=today)
        row = ObligationChecklistItem(
            user_id=user_id,
            title=item.title,
            category=item.category,
            description=item.description,
            deadline_type=item.deadline_rule.type,
            deadline_month=item.deadline_rule.month,
            deadline_day=item.deadline_rule.day,
            period_months=item.deadline_rule.period_months,
            days_after_event=item.deadline_rule.days_after_event,
            event_description=item.deadline_rule.event_description,
            due_date=due,
            penalty_summary=item.penalty_summary,
            source_citation=item.source_citation,
            source_chunk_id=item.source_chunk_id,
            status="pending",
        )
        db.add(row)
        saved.append(row)

    db.commit()
    for row in saved:
        db.refresh(row)

    logger.info("Checklist generation for user %s: %d obligation(s) saved", user_id, len(saved))
    return saved
