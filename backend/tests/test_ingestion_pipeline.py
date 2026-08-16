"""Integration tests for the full ingestion pipeline (pipeline.py +
vector_store.py), using chromadb's in-memory EphemeralClient — never a
real Chroma server, and never a real embedding model (the
HashingEmbeddingProvider is used throughout, since these tests only
need to verify PLUMBING correctness — chunk counts, idempotency,
diffing, query wiring — not retrieval quality).

These tests directly verify the Phase 2 Definition of Done from
docs/IMPLEMENTATION_PLAN.md:
  - running ingestion populates the vector DB with correctly-bounded chunks
  - re-running it on unchanged source docs makes no writes
  - running it after editing one source doc only re-embeds that doc
"""
from __future__ import annotations

import uuid

import chromadb
import pytest

from app.ingestion.chunker import chunk_document
from app.ingestion.embeddings import HashingEmbeddingProvider
from app.ingestion.metadata import SourceMeta, build_metadata
from app.ingestion.parser import parse_document
from app.ingestion.pipeline import ingest_document
from app.ingestion.vector_store import VectorStoreWriter

SAMPLE_TEXT = """\
Chương I
HEADING

Điều 1. First article
1. Some content for article one.

Điều 2. Second article
1. Some content for article two.
2. More content for article two.
"""

TEST_SOURCE = SourceMeta(law_name="Test Fixture Law", category="test_fixture", entity_type="both")


@pytest.fixture()
def store() -> VectorStoreWriter:
    # Unique collection name per test — see the warning in
    # app/ingestion/vector_store.py: separate EphemeralClient()
    # instances can otherwise share storage by collection name.
    client = chromadb.EphemeralClient()
    return VectorStoreWriter(client=client, collection_name=f"test-{uuid.uuid4().hex}")


@pytest.fixture()
def embedder() -> HashingEmbeddingProvider:
    return HashingEmbeddingProvider()


def test_ingestion_writes_one_chunk_per_article(store, embedder):
    result = ingest_document(
        "doc1", "Doc One", SAMPLE_TEXT, TEST_SOURCE, vector_store=store, embedder=embedder
    )
    assert result["article_count"] == 2
    assert result["total"] == 2
    assert result["written"] == 2
    assert result["skipped"] == 0
    assert store.count() == 2


def test_chunk_metadata_is_queryable(store, embedder):
    ingest_document("doc1", "Doc One", SAMPLE_TEXT, TEST_SOURCE, vector_store=store, embedder=embedder)
    chunk = store.get_chunk("doc1:dieu1")
    assert chunk is not None
    assert chunk["metadata"]["article_number"] == "1"
    assert chunk["metadata"]["category"] == "test_fixture"
    assert "Điều 1" in chunk["text"]


def test_reingesting_unchanged_document_writes_nothing(store, embedder):
    ingest_document("doc1", "Doc One", SAMPLE_TEXT, TEST_SOURCE, vector_store=store, embedder=embedder)
    result = ingest_document(
        "doc1", "Doc One", SAMPLE_TEXT, TEST_SOURCE, vector_store=store, embedder=embedder
    )
    assert result["written"] == 0
    assert result["skipped"] == 2
    assert store.count() == 2  # no duplicates either


def test_editing_one_article_only_rewrites_that_articles_chunk(store, embedder):
    ingest_document("doc1", "Doc One", SAMPLE_TEXT, TEST_SOURCE, vector_store=store, embedder=embedder)

    edited_text = SAMPLE_TEXT.replace(
        "1. Some content for article one.", "1. COMPLETELY DIFFERENT content for article one now."
    )
    result = ingest_document(
        "doc1", "Doc One", edited_text, TEST_SOURCE, vector_store=store, embedder=embedder
    )

    assert result["written"] == 1  # only Điều 1 changed
    assert result["skipped"] == 1  # Điều 2 untouched
    assert store.count() == 2  # still just 2 chunks total, not 3

    updated_chunk = store.get_chunk("doc1:dieu1")
    assert "COMPLETELY DIFFERENT" in updated_chunk["text"]
    untouched_chunk = store.get_chunk("doc1:dieu2")
    assert "COMPLETELY DIFFERENT" not in untouched_chunk["text"]


def test_query_returns_the_most_similar_stored_chunk(store, embedder):
    ingest_document("doc1", "Doc One", SAMPLE_TEXT, TEST_SOURCE, vector_store=store, embedder=embedder)

    query_vec = embedder.embed(["content for article one"])[0]
    result = store.query(query_vec, n_results=1)

    assert result["ids"][0][0] == "doc1:dieu1"


def test_delete_document_removes_all_its_chunks(store, embedder):
    ingest_document("doc1", "Doc One", SAMPLE_TEXT, TEST_SOURCE, vector_store=store, embedder=embedder)
    assert store.count() == 2

    store.delete_document("doc1")
    assert store.count() == 0


def test_ingesting_two_different_documents_keeps_them_independent(store, embedder):
    ingest_document("doc1", "Doc One", SAMPLE_TEXT, TEST_SOURCE, vector_store=store, embedder=embedder)
    other_text = "Điều 1. Unrelated\n1. Unrelated content entirely.\n"
    ingest_document("doc2", "Doc Two", other_text, TEST_SOURCE, vector_store=store, embedder=embedder)

    assert store.count() == 3  # 2 from doc1 + 1 from doc2
    store.delete_document("doc1")
    assert store.count() == 1  # only doc2's chunk remains


def _batch_items(doc_id, title, text, source=TEST_SOURCE):
    document = parse_document(doc_id, title, text)
    return [(chunk, build_metadata(chunk, source)) for chunk in chunk_document(document)]


def test_upsert_chunks_batched_writes_all_and_is_idempotent(store, embedder):
    items = _batch_items("doc1", "Doc One", SAMPLE_TEXT)
    first = store.upsert_chunks_batched(items, embedder, flush_size=1)
    assert first["total"] == 2
    assert first["written"] == 2
    assert first["skipped"] == 0
    assert store.count() == 2

    # Same items again — every chunk's content hash already matches, so
    # the batched dedup `get()` must skip all of them, exactly like the
    # per-chunk upsert_chunks path.
    second = store.upsert_chunks_batched(items, embedder, flush_size=1)
    assert second["written"] == 0
    assert second["skipped"] == 2
    assert store.count() == 2


def test_upsert_chunks_batched_mixes_new_and_existing(store, embedder):
    items1 = _batch_items("doc1", "Doc One", SAMPLE_TEXT)
    store.upsert_chunks_batched(items1, embedder)

    other_text = "Điều 1. Unrelated\n1. Unrelated content entirely.\n"
    items2 = _batch_items("doc2", "Doc Two", other_text)

    # doc1's chunks are already stored; only doc2's single chunk is new.
    mixed = items1 + items2
    result = store.upsert_chunks_batched(mixed, embedder, flush_size=1)
    assert result["written"] == 1
    assert result["skipped"] == 2
    assert store.count() == 3
