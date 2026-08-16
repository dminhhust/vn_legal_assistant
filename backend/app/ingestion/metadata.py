"""Metadata tagging for ingested legal chunks.

Category taxonomy matches docs/ARCHITECTURE.md §3.1 — stored as a
plain list of strings (config data), not a hardcoded enum, so adding a
new category is a data change, never a code change.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from app.ingestion.chunker import Chunk

CATEGORIES = [
    "tax",
    "contracts_signing",
    "labor_insurance",
    "residence_civil",
    "business_licensing",
    "property_vehicles",
    "family_civil",
    # Reserved for pipeline self-tests only — never a real obligation
    # category, and filtered out of any real retrieval query.
    "test_fixture",
]
ENTITY_TYPES = ["individual", "business", "both"]


@dataclass
class SourceMeta:
    """Document-level metadata supplied by whoever registers a source
    with the ingestion pipeline (see run_ingestion.py)."""

    law_name: str
    category: str  # one of CATEGORIES
    entity_type: str = "both"  # one of ENTITY_TYPES
    province_scope: Optional[str] = None  # None = national; else a specific province
    effective_from: Optional[str] = None  # ISO date string, e.g. "2024-01-01"
    effective_to: Optional[str] = None  # ISO date string; None/empty = still in effect
    source_url: Optional[str] = None


def content_hash(text: str) -> str:
    """Stable hash used for idempotent re-ingestion — see
    vector_store.py's upsert_chunks, which skips writing any chunk
    whose stored hash already matches."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_metadata(chunk: Chunk, source: SourceMeta) -> dict:
    if source.category not in CATEGORIES:
        raise ValueError(f"Unknown category '{source.category}'. Add it to CATEGORIES first.")
    if source.entity_type not in ENTITY_TYPES:
        raise ValueError(f"Unknown entity_type '{source.entity_type}'. Must be one of {ENTITY_TYPES}.")

    return {
        "document_id": chunk.document_id,
        "document_title": chunk.document_title,
        "law_name": source.law_name,
        "category": source.category,
        "entity_type": source.entity_type,
        "province_scope": source.province_scope or "national",
        "effective_from": source.effective_from or "",
        "effective_to": source.effective_to or "",
        "chapter_number": chunk.chapter_number or "",
        "chapter_title": chunk.chapter_title or "",
        "article_number": chunk.article_number,
        "article_title": chunk.article_title,
        "part_index": chunk.part_index,
        "part_count": chunk.part_count,
        "source_url": source.source_url or "",
        "content_hash": content_hash(chunk.text),
    }
