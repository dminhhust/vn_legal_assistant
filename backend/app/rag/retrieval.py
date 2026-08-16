"""Hybrid retrieval (docs/ARCHITECTURE.md §4.2): BM25 (lexical) + vector
similarity, fused via Reciprocal Rank Fusion (RRF), filtered by entity
type (individual/business) and province jurisdiction. This is the
pipeline actually wired into the live app — app/chat/tools.py's
`search_legal_obligations`, app/rag/checklist_service.py's
`generate_checklist_for_user`, and (indirectly, via `RetrievedChunk`)
app/rag/extraction.py and app/rag/reranker.py all import from here.

Two signals, not one, because they fail differently:
  - BM25 (`rank_bm25.BM25Okapi`) is exact-term sensitive — it won't
    find a paraphrase, but it's robust regardless of embedding
    quality, which matters a lot here: this app's default embedding
    fallback, `HashingEmbeddingProvider`
    (app/ingestion/embeddings.py), is explicitly NOT semantically
    meaningful.
  - Vector similarity (cosine, over whatever `EmbeddingProvider` is
    configured) finds paraphrases/synonyms a pure keyword match would
    miss, but only when a real embedding model is actually configured.
Fusing both via RRF means retrieval quality degrades gracefully to
"BM25 only, effectively" when no real embedding provider is set,
rather than depending entirely on a signal that may not be
semantically meaningful in this deployment.

Candidate generation is category-scoped, not corpus-wide: every call
to `retrieve()` first pulls the FULL set of chunks tagged with the
query's `category` via `VectorStoreWriter.get_by_metadata` (see that
method's own docstring — this is exactly what it exists for: the full
candidate set BM25 needs, not just a top-k vector result), then
applies the entity_type/province jurisdiction filter, THEN scores what
survives with both signals. Filtering before scoring (rather than
scoring everything and filtering after) keeps BM25's corpus scoped to
what could actually be returned, and keeps a filtered-out chunk from
ever occupying a rank slot the fusion step would otherwise give a
chunk that's actually in scope for this user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import unicodedata

from rank_bm25 import BM25Okapi

from app.ingestion.embeddings import EmbeddingProvider
from app.ingestion.vector_store import VectorStoreWriter
from app.rag.query_builder import CategoryQuery

# The standard RRF constant from Cormack, Clarke & Buettcher (2009),
# "Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank
# Learning Methods" — chosen so a single high-ranked hit from one
# signal alone doesn't dominate the fused order over a chunk that
# ranks well on BOTH signals.
_RRF_K = 60
_DEFAULT_TOP_N_PER_SIGNAL = 5

# When the configured embedding provider is genuinely semantic (see
# embeddings.py's `is_semantic` flag), retrieve() bounds the candidate
# set BM25 scores with a vector-pre-filtered pool instead of pulling
# the FULL category into memory. This is what keeps retrieval usable
# at full-corpus scale: after ingesting the entire ~158K-doc vbpl-vn
# dataset a single category can hold tens of thousands of chunks, and
# scoring all of them with BM25 per query is seconds of CPU + a huge
# Chroma fetch per category. Pooling the vector top-k first caps both
# at `_DEFAULT_CANDIDATE_POOL_SIZE` regardless of category size. BM25
# still contributes real signal — it re-ranks within the pool and is
# fused with the vector order via RRF — but can no longer surface a
# purely-lexical hit that the semantic filter ranked far down. That
# trade is only acceptable when the vector signal is real, which is
# exactly why the full-category path is kept for hashing (non-semantic)
# embedders: for them BM25 IS the only signal and must see everything.
_DEFAULT_CANDIDATE_POOL_SIZE = 100


@dataclass
class RetrievedChunk:
    """One retrieval hit, ready for reranking/extraction. `fused_score`
    defaults to 0.0 (not required at construction — see
    tests/test_rag_reranker.py and tests/test_rag_extraction.py, which
    build these directly without it) since it's meaningful only for a
    chunk that actually came out of `HybridRetriever.retrieve()`.

    `bm25_score` is the RAW BM25 score for the query, carried alongside
    the fused score. RRF fused scores collapse to a few discrete values
    (1/(60+k) sums — typically 0.016/0.032), so comparing fused scores
    ACROSS separate `retrieve()` calls (different categories) is a
    lottery among ties — that is exactly how chat's cross-category
    ranking mis-picked off-topic fallback chunks (found by measuring
    retrieval after the query/tokenizer fixes). Callers that compare
    hits from multiple category scopes should break ties with the raw
    lexical score, which is a real graded signal in this codebase's
    documented "degrade to BM25" mode (see the module docstring)."""

    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)
    fused_score: float = 0.0
    bm25_score: float = 0.0


def _strip_diacritics(token: str) -> str:
    """NFD-normalizes and removes combining marks — the standard
    approach for making Vietnamese text comparable with or without
    tone marks ("thuế" -> "thue"). Used by _tokenize so BM25 matches
    regardless of whether the query and the corpus spell a word with
    diacritics (the corpus always does; users typing in chat often
    don't)."""
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _tokenize(text: str) -> list[str]:
    """Whitespace tokenization, lowercased, plus a diacritic-stripped
    variant of every token. Vietnamese orthography already space-
    separates syllables, so this is a reasonable, simple BM25 tokenizer
    without needing a dedicated Vietnamese word segmenter — matching
    the level of simplicity this codebase already uses elsewhere for
    keyword-ish text processing. Appending the diacritic-stripped
    forms is what lets an un-accented query match an accented corpus
    and vice versa (found by actually measuring retrieval: English
    templates + exact-term matching made BM25 score nothing against
    Vietnamese chunks — see docs/PERFORMANCE_ANALYSIS.md §4.2)."""
    tokens = text.lower().split()
    stripped = [_strip_diacritics(t) for t in tokens]
    out = list(tokens)
    for t in stripped:
        if t and t not in tokens:
            out.append(t)
    return out


def _matches_entity_type(metadata: dict, user_entity_type: str) -> bool:
    chunk_entity_type = metadata.get("entity_type", "both")
    return chunk_entity_type in (user_entity_type, "both")


def _matches_province(metadata: dict, user_province: Optional[str]) -> bool:
    """`province_scope` is `"national"` (always in scope) or a specific
    province name (in scope only for a user who has told us that same
    province — a user with no province on file sees national law
    only, never guessed provincial law)."""
    province_scope = metadata.get("province_scope", "national")
    if province_scope == "national":
        return True
    return user_province is not None and province_scope == user_province


def _rrf_fuse(*ranked_id_lists: list[str]) -> dict[str, float]:
    """Reciprocal Rank Fusion over any number of already-ranked
    (best-first) chunk_id lists. A chunk_id absent from a given list
    contributes 0 for that list — it isn't penalized beyond simply not
    getting that list's contribution."""
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank)
    return scores


class HybridRetriever:
    """`vector_store` is a `VectorStoreWriter` (Chroma-backed in
    production, `EphemeralClient`-backed in tests); `embedder` is
    whatever `EmbeddingProvider` the caller has configured. Both are
    plain constructor args (not keyword-only) to match every existing
    call site — see app/chat/tools.py and
    app/rag/checklist_service.py."""

    def __init__(self, vector_store: VectorStoreWriter, embedder: EmbeddingProvider):
        self._store = vector_store
        self._embedder = embedder

    def retrieve(
        self,
        category_query: CategoryQuery,
        *,
        user_entity_type: str,
        user_province: Optional[str],
        top_n_per_signal: int = _DEFAULT_TOP_N_PER_SIGNAL,
    ) -> list[RetrievedChunk]:
        """Returns the fused, best-first list of matching chunks for
        one category query. `top_n_per_signal` is exactly what it
        says: the number of top candidates taken FROM EACH signal
        (BM25 top-N, vector top-N) before fusion — so the result can
        contain up to `2 * top_n_per_signal` unique chunks (fewer
        after de-dup, or after province/entity_type filtering removes
        some of a signal's raw hits), never a hard cap at
        `top_n_per_signal` itself. Callers that want a smaller final
        count slice the result themselves (see app/chat/tools.py,
        which caps at 2 per category after calling this with
        `top_n_per_signal=5`).

        Candidate generation depends on the embedding provider:

          - `is_semantic == True` (real models: OpenAI/Gemini/local
            sentence-transformers): the candidate pool is the vector
            top-`_DEFAULT_CANDIDATE_POOL_SIZE` per category (see
            `_retrieve_pruned`). Bounded regardless of category size —
            required for the full-corpus index.
          - non-semantic (HashingEmbeddingProvider): the FULL category
            is pulled and BM25 scores everything (see `_retrieve_full`).
            Here BM25 is the only meaningful signal, so it must see the
            whole set. This is also the path all retrieval unit tests
            exercise, since they use the hashing embedder.

        Returns `[]` — never raises — when nothing in this category
        exists yet, or nothing survives the jurisdiction filter; both
        are routine (an unseeded corpus, a user with no matching
        provincial law), not error conditions."""
        if getattr(self._embedder, "is_semantic", False):
            return self._retrieve_pruned(
                category_query,
                user_entity_type=user_entity_type,
                user_province=user_province,
                top_n_per_signal=top_n_per_signal,
            )
        return self._retrieve_full(
            category_query,
            user_entity_type=user_entity_type,
            user_province=user_province,
            top_n_per_signal=top_n_per_signal,
        )

    def _retrieve_full(
        self,
        category_query: CategoryQuery,
        *,
        user_entity_type: str,
        user_province: Optional[str],
        top_n_per_signal: int,
    ) -> list[RetrievedChunk]:
        """The original candidate-generation strategy: pull the ENTIRE
        category, filter for jurisdiction, BM25-score everything. Kept
        for non-semantic embedders and exercised by the unit tests —
        see `retrieve()`'s docstring for why it must stay."""
        all_in_category = self._store.get_by_metadata({"category": category_query.category})
        filtered = [
            c
            for c in all_in_category
            if _matches_entity_type(c["metadata"], user_entity_type)
            and _matches_province(c["metadata"], user_province)
        ]
        if not filtered:
            return []

        by_id = {c["id"]: c for c in filtered}
        query_text = category_query.query_text

        bm25_ranked = self._bm25_rank(query_text, filtered, top_n_per_signal)
        bm25_ids = [chunk_id for chunk_id, _ in bm25_ranked]
        bm25_by_id = dict(bm25_ranked)
        vector_ranked_ids = self._vector_rank(
            query_text, category_query.category, top_n_per_signal, len(all_in_category), by_id
        )

        fused = _rrf_fuse(bm25_ids, vector_ranked_ids)
        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=by_id[chunk_id]["text"],
                metadata=by_id[chunk_id]["metadata"],
                fused_score=score,
                bm25_score=bm25_by_id.get(chunk_id, 0.0),
            )
            for chunk_id, score in ordered
        ]

    def _retrieve_pruned(
        self,
        category_query: CategoryQuery,
        *,
        user_entity_type: str,
        user_province: Optional[str],
        top_n_per_signal: int,
        candidate_pool_size: int = _DEFAULT_CANDIDATE_POOL_SIZE,
    ) -> list[RetrievedChunk]:
        """Vector-pre-filtered candidate pool (see `retrieve()`). One
        Chroma `query` call returns the top `candidate_pool_size`
        chunks for the category in vector-similarity order (never more
        than exist — Chroma returns what's there), the jurisdiction
        filter is applied to that pool in Python, and BM25 re-ranks
        the survivors. The vector signal IS the pool order; fusion is
        RRF over (BM25 top-N, pool order). Cost is O(pool_size) no
        matter how large the category grew, which is the whole point —
        after a full vbpl-vn ingest a category can hold ~10^4 chunks
        and the pool caps both the Chroma transfer and the BM25
        scoring."""
        query_text = category_query.query_text
        category = category_query.category
        query_embedding = self._embedder.embed([query_text], task_type="RETRIEVAL_QUERY")[0]
        raw = self._store.query(
            query_embedding, n_results=candidate_pool_size, where={"category": category}
        )
        ids = raw.get("ids") or [[]]
        docs = raw.get("documents") or [[]]
        metas = raw.get("metadatas") or [[]]
        candidates = [
            {"id": ids[0][i], "text": docs[0][i], "metadata": metas[0][i]}
            for i in range(len(ids[0]))
        ]
        filtered = [
            c
            for c in candidates
            if _matches_entity_type(c["metadata"], user_entity_type)
            and _matches_province(c["metadata"], user_province)
        ]
        if not filtered:
            return []

        by_id = {c["id"]: c for c in filtered}
        # Pool order is the vector ranking (best-first), so the vector
        # signal is `filtered`'s own order — the jurisdiction filter
        # only removes members, never reorders.
        vector_ids = [c["id"] for c in filtered]

        bm25_ranked = self._bm25_rank(query_text, filtered, top_n_per_signal)
        bm25_ids = [chunk_id for chunk_id, _ in bm25_ranked]
        bm25_by_id = dict(bm25_ranked)

        fused = _rrf_fuse(bm25_ids, vector_ids)
        ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=by_id[chunk_id]["text"],
                metadata=by_id[chunk_id]["metadata"],
                fused_score=score,
                bm25_score=bm25_by_id.get(chunk_id, 0.0),
            )
            for chunk_id, score in ordered
        ]

    def _bm25_rank(self, query_text: str, filtered: list[dict], top_n: int) -> list[tuple[str, float]]:
        corpus = [_tokenize(c["text"]) for c in filtered]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(_tokenize(query_text))
        ranked = sorted(zip(filtered, scores), key=lambda pair: pair[1], reverse=True)
        return [(c["id"], float(score)) for c, score in ranked[:top_n]]

    def _vector_rank(
        self,
        query_text: str,
        category: str,
        top_n: int,
        category_size: int,
        by_id: dict[str, dict],
    ) -> list[str]:
        """Queries Chroma scoped to `category` only (Chroma's `where`
        stays a single simple equality filter here rather than
        depending on version-specific `$in`/`$or` support for
        entity_type/province); the jurisdiction filter is re-applied
        in Python against `by_id` afterward — same
        "store applies what it can efficiently, caller re-checks
        the real filter" belt-and-suspenders pattern used elsewhere in
        this codebase (see app/rag/obligation_retrieval.py's
        `_candidates` for the equivalent in the other retrieval
        pipeline). `n_results` is capped at `category_size` so this
        never asks Chroma for more results than exist in the
        category, which some Chroma versions reject outright."""
        n_results = min(top_n, category_size)
        if n_results <= 0:
            return []
        query_embedding = self._embedder.embed([query_text], task_type="RETRIEVAL_QUERY")[0]
        raw = self._store.query(query_embedding, n_results=n_results, where={"category": category})
        ids = (raw.get("ids") or [[]])[0]
        # Keep only ids that survived the entity_type/province filter
        # (by_id is already that filtered set) and preserve Chroma's
        # own best-first order.
        return [chunk_id for chunk_id in ids if chunk_id in by_id]
