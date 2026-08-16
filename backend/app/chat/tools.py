"""Tool implementations for the RAG chatbot. Each function is plain,
testable Python — no LLM involved.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import ObligationChecklistItem, User
from app.ingestion.embeddings import EmbeddingProvider, get_default_embedding_provider
from app.ingestion.metadata import CATEGORIES
from app.ingestion.vector_store import VectorStoreWriter
from app.rag.query_builder import CategoryQuery, infer_entity_type
from app.rag.retrieval import HybridRetriever

_REAL_CATEGORIES = [c for c in CATEGORIES if c != "test_fixture"]


def search_legal_obligations(
    query: str,
    *,
    user_id: str,
    db: Session,
    vector_store: Optional[VectorStoreWriter] = None,
    embedder: Optional[EmbeddingProvider] = None,
) -> str:
    """Ad-hoc retrieval for a free-text question — distinct from the
    checklist generator's category-scoped BATCH queries. Searches
    across every real category (not just ones the checklist generator
    proactively surfaced for this user), since an ad-hoc question might
    reasonably be about anything.

    NOTE: this loops retrieval once per category, which is simple and
    correct but not the most efficient shape for a large real corpus —
    a single un-scoped vector search with category as a soft signal
    would scale better. Flagged as a known optimization opportunity for
    post-MVP, not a blocker for this MVP's scope.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        return "No profile found for this user."

    trait_tags = [t.tag for t in user.traits]
    entity_type = infer_entity_type(trait_tags)
    province = user.profile.province if user.profile else None

    store = vector_store or VectorStoreWriter()
    embed_provider = embedder or get_default_embedding_provider()
    retriever = HybridRetriever(store, embed_provider)

    all_hits = []
    for category in _REAL_CATEGORIES:
        cq = CategoryQuery(category=category, query_text=query, matched_traits=[])
        hits = retriever.retrieve(
            cq, user_entity_type=entity_type, user_province=province, top_n_per_signal=5
        )
        all_hits.extend(hits[:2])  # cap per category so context doesn't explode

    if not all_hits:
        return "No relevant legal information found in the knowledge base."

    # Fused scores cluster into a few discrete RRF values (0.016/0.032),
    # so a cross-category sort on fused_score alone is a lottery among
    # ties — it mis-picked off-topic fallback chunks when this was
    # measured against the real corpus. Break ties with the raw BM25
    # score, the one genuinely graded lexical signal available in this
    # deployment (see RetrievedChunk.bm25_score's docstring).
    all_hits.sort(key=lambda h: (h.fused_score, h.bm25_score), reverse=True)
    top_hits = all_hits[:5]

    blocks = []
    for hit in top_hits:
        law_name = hit.metadata.get("law_name", "unknown source")
        article_number = hit.metadata.get("article_number", "?")
        blocks.append(f"Source: {law_name}, Điều {article_number}\n{hit.text}")
    return "\n\n---\n\n".join(blocks)


def get_checklist_status(days_ahead: Optional[int], *, user_id: str, db: Session) -> str:
    items = (
        db.query(ObligationChecklistItem)
        .filter(ObligationChecklistItem.user_id == user_id, ObligationChecklistItem.status == "pending")
        .all()
    )
    if days_ahead is not None:
        cutoff = date.today() + timedelta(days=days_ahead)
        items = [i for i in items if i.due_date is not None and i.due_date <= cutoff]

    if not items:
        suffix = f" due within {days_ahead} days" if days_ahead is not None else ""
        return f"No pending checklist items{suffix}."

    lines = [
        f"- [{i.category}] {i.title} — due {i.due_date} — {i.source_citation}" for i in items
    ]
    return "\n".join(lines)


def mark_checklist_item_done(title_contains: str, *, user_id: str, db: Session) -> str:
    matches = (
        db.query(ObligationChecklistItem)
        .filter(
            ObligationChecklistItem.user_id == user_id,
            ObligationChecklistItem.title.ilike(f"%{title_contains}%"),
        )
        .all()
    )
    if not matches:
        return f"No checklist item found matching '{title_contains}'."
    if len(matches) > 1:
        titles = "; ".join(m.title for m in matches)
        return f"Multiple items match '{title_contains}': {titles}. Please be more specific."

    matches[0].status = "done"
    db.commit()
    return f"Marked '{matches[0].title}' as done."
