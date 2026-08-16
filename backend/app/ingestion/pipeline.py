"""Orchestrates the full ingestion flow: parse -> chunk -> tag -> embed
-> write. See docs/ARCHITECTURE.md §4.5 for the design rationale behind
each step.
"""
from __future__ import annotations

import logging
from typing import Optional

from app.ingestion.chunker import DEFAULT_MAX_CHARS, chunk_document
from app.ingestion.embeddings import EmbeddingProvider, get_default_embedding_provider
from app.ingestion.metadata import SourceMeta
from app.ingestion.parser import parse_document
from app.ingestion.vector_store import VectorStoreWriter

logger = logging.getLogger(__name__)


def ingest_document(
    doc_id: str,
    title: str,
    raw_text: str,
    source: SourceMeta,
    *,
    vector_store: Optional[VectorStoreWriter] = None,
    embedder: Optional[EmbeddingProvider] = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict:
    """Runs one document through the full pipeline and returns a result
    summary (written/skipped/total chunk counts + article count).
    Idempotent — see VectorStoreWriter.upsert_chunks."""
    document = parse_document(doc_id, title, raw_text)
    chunks = chunk_document(document, max_chars=max_chars)

    store = vector_store or VectorStoreWriter()
    embed_provider = embedder or get_default_embedding_provider()

    result = store.upsert_chunks(chunks, source, embed_provider)
    logger.info(
        "Ingested '%s': %d article(s) -> %d chunk(s) (%d written, %d unchanged)",
        title,
        len(document.all_articles()),
        result["total"],
        result["written"],
        result["skipped"],
    )
    return {**result, "document_id": doc_id, "article_count": len(document.all_articles())}
