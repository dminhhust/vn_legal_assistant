"""Embeddings have to be computed ourselves — `tmquan/vbpl-vn` ships
the documents parquet only, no embeddings column. This module owns
exactly two responsibilities: normalizing text the same way for every
document before embedding it, and defining the provider interface
obligation_retrieval.py depends on (never a concrete model call directly), so the
pipeline is testable offline the same way this codebase already tests
its HTTP clients and LLM router — real provider swapped for a fixture
one in tests, same call shape either way.

MODEL: the dataset card's own choice, nvidia/llama-nemotron-embed-1b-v2,
is used here for embedding computation specifically so retrieval
similarity is comparable to how the dataset's authors validated their
own corpus (a different embedding model is not guaranteed to produce
similar nearest-neighbour behaviour). Running that model is out of
scope for this module — it's a real inference workload (GPU-hosted or
via an inference API) that belongs in a separate ingestion job, not
inline in a request-time retrieval call. `Embedder` below is the
Protocol that job's real implementation and this module's tests both
satisfy.
"""
from __future__ import annotations

import unicodedata
from typing import Protocol, Sequence

EMBEDDING_MODEL_NAME = "nvidia/llama-nemotron-embed-1b-v2"


def normalize_for_embedding(markdown: str) -> str:
    """NFC-normalize before embedding — Vietnamese text can arrive in
    either NFC or NFD (combining diacritics) depending on the
    producing tool, and the two encode the same visible text as
    different byte/codepoint sequences. Embedding un-normalized text
    would put visually-identical Vietnamese strings at different
    points in the vector space depending on which normalization form
    happened to produce them, which is a silent, hard-to-debug recall
    bug. Every call site in this package (indexing AND query-time
    embedding) must go through this function so both sides of a
    similarity comparison are normalized identically."""
    return unicodedata.normalize("NFC", markdown)


class Embedder(Protocol):
    """Provider abstraction — same pattern as this codebase's
    HttpClient/LlmRouter injection. A real implementation calls
    nvidia/llama-nemotron-embed-1b-v2 (self-hosted or via an inference
    API); tests substitute a deterministic fixture embedder."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Texts MUST already be NFC-normalized by the caller (via
        `normalize_for_embedding`) — this Protocol does not normalize
        for you, so indexing and query-time embedding stay obviously
        symmetric at the call site rather than symmetric by
        coincidence."""
        ...

    def embed_query(self, text: str) -> list[float]:
        ...
