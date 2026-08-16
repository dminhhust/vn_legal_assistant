"""Vector DB writer — wraps a Chroma client with idempotent upsert
logic based on content-hash diffing (see docs/ARCHITECTURE.md §4.5:
"Incremental updates").

In real deployments this talks to Chroma over HTTP, matching
docker-compose's standalone `chroma` service (app.config.CHROMA_HOST /
CHROMA_PORT). Tests use chromadb's in-memory EphemeralClient instead —
see tests/test_ingestion_pipeline.py — and this class always passes
pre-computed embeddings explicitly (never Chroma's built-in embedding
function), so nothing here ever needs to download an embedding model.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import chromadb

from app.config import CHROMA_HOST, CHROMA_PORT
from app.ingestion.chunker import Chunk
from app.ingestion.embeddings import EmbeddingProvider
from app.ingestion.metadata import SourceMeta, build_metadata

logger = logging.getLogger(__name__)

COLLECTION_NAME = "legal_corpus"

# IMPORTANT — discovered via testing, not from Chroma's docs: separate
# chromadb.EphemeralClient() instances in the same process can share
# underlying storage keyed by collection NAME, even though the client
# objects themselves are distinct. This bit test isolation the first
# time this was tried (Phase 3 test suite showed chunk counts far
# higher than expected, traced to leakage across unrelated tests' data
# under the same default collection name). Callers that need genuine
# isolation (every test fixture in this codebase) MUST pass a unique
# `collection_name` — see tests/test_ingestion_pipeline.py,
# tests/test_rag_retrieval.py, tests/test_checklist_service.py.


class VectorStoreWriter:
    def __init__(self, client: Optional[Any] = None, collection_name: str = COLLECTION_NAME):
        self._client = client or chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert_chunks(
        self, chunks: list[Chunk], source: SourceMeta, embedder: EmbeddingProvider
    ) -> dict:
        """Idempotent: a chunk whose content_hash already matches what's
        stored is skipped entirely — no re-embedding, no write. A chunk
        whose hash changed (or that's new) is (re-)written. Returns
        counts so callers/tests can assert on exactly what happened.
        """
        to_write_ids: list[str] = []
        to_write_chunks: list[Chunk] = []
        to_write_meta: list[dict] = []
        skipped = 0

        for chunk in chunks:
            meta = build_metadata(chunk, source)
            existing = self._get_existing_metadata(chunk.chunk_id)
            if existing is not None and existing.get("content_hash") == meta["content_hash"]:
                skipped += 1
                continue
            to_write_ids.append(chunk.chunk_id)
            to_write_chunks.append(chunk)
            to_write_meta.append(meta)

        if to_write_ids:
            embeddings = embedder.embed([c.text for c in to_write_chunks])
            self._collection.upsert(
                ids=to_write_ids,
                embeddings=embeddings,
                documents=[c.text for c in to_write_chunks],
                metadatas=to_write_meta,
            )

        result = {"written": len(to_write_ids), "skipped": skipped, "total": len(chunks)}
        logger.info("upsert_chunks: %s", result)
        return result

    def _get_existing_metadata(self, chunk_id: str) -> Optional[dict]:
        result = self._collection.get(ids=[chunk_id], include=["metadatas"])
        metadatas = result.get("metadatas") or []
        return metadatas[0] if metadatas else None

    def upsert_chunks_batched(
        self, items: list[tuple[Chunk, dict]], embedder: EmbeddingProvider, *, flush_size: int = 500
    ) -> dict:
        """Batched idempotent upsert for full-corpus ingests. Takes a
        list of `(chunk, metadata)` pairs (metadata from
        `build_metadata(chunk, source)` — see metadata.py) and writes
        them in one dedup `get()`, one embedding call, and one upsert
        per flush.

        WHY THIS EXISTS: `upsert_chunks` issues a Chroma HTTP `get()`
        PER CHUNK for the content-hash dedup check. That's fine for a
        handful of documents, but the full vbpl-vn dataset is ~158K
        documents / ~500K+ chunks — 500K serial HTTP round-trips is
        hours of wall time and thousands of Chroma requests. Batching
        makes the whole corpus a few hundred round-trips instead (see
        docs/DEBUGGING_LOOP_LOG.md Iteration 11). Idempotency is
        preserved: the batch `get(ids=...)` returns existing metadata
        for whatever is already stored, so already-ingested chunks are
        skipped exactly like in `upsert_chunks`."""
        written = 0
        skipped = 0
        total = len(items)

        pending = list(items)
        while pending:
            batch = pending[:flush_size]
            pending = pending[flush_size:]

            ids = [chunk.chunk_id for chunk, _ in batch]
            existing = self._collection.get(ids=ids, include=["metadatas"])
            existing_hashes = {
                existing["ids"][i]: (existing["metadatas"][i] or {}).get("content_hash")
                for i in range(len(existing.get("ids") or []))
            }

            to_write = [
                (chunk, meta)
                for chunk, meta in batch
                if existing_hashes.get(chunk.chunk_id) != meta["content_hash"]
            ]
            skipped += len(batch) - len(to_write)

            if to_write:
                chunk_ids = [c.chunk_id for c, _ in to_write]
                texts = [c.text for c, _ in to_write]
                metadatas = [meta for _, meta in to_write]
                embeddings = embedder.embed(texts)
                self._collection.upsert(
                    ids=chunk_ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                written += len(to_write)

        logger.info(
            "upsert_chunks_batched: %d total (%d written, %d skipped)",
            total,
            written,
            skipped,
        )
        return {"written": written, "skipped": skipped, "total": total}

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        result = self._collection.get(ids=[chunk_id], include=["metadatas", "documents"])
        if not result.get("ids"):
            return None
        return {
            "id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    def get_by_metadata(self, where: dict) -> list[dict]:
        """Returns EVERY stored chunk matching a metadata filter — the
        full candidate set a hybrid retriever needs for keyword (BM25)
        scoring, not just a top-k vector result. Chroma's `get()`
        supports this without needing a query embedding at all."""
        result = self._collection.get(where=where, include=["documents", "metadatas"])
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        return [
            {"id": chunk_id, "text": docs[i], "metadata": metas[i]}
            for i, chunk_id in enumerate(ids)
        ]

    def count(self) -> int:
        return self._collection.count()

    def query(
        self, query_embedding: list[float], n_results: int = 5, where: Optional[dict] = None
    ) -> dict:
        return self._collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=where
        )

    def delete_document(self, document_id: str) -> None:
        """Removes all chunks belonging to one source document — used
        when a law is superseded/withdrawn and needs to be pulled
        rather than just left stale in the index."""
        self._collection.delete(where={"document_id": document_id})
