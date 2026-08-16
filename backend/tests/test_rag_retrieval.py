"""Tests for app/rag/retrieval.py's `HybridRetriever` — BM25 + vector
similarity fused via RRF, filtered by entity_type + province. Uses a
real in-memory Chroma (`EphemeralClient`) and the offline
`HashingEmbeddingProvider`, same pattern as
tests/test_checklist_service.py and tests/test_chat_tools.py: the
plumbing is what's under test, not embedding semantics.
"""
from __future__ import annotations

import uuid

import chromadb
import pytest

from app.ingestion.embeddings import HashingEmbeddingProvider
from app.ingestion.metadata import SourceMeta
from app.ingestion.pipeline import ingest_document
from app.ingestion.vector_store import VectorStoreWriter
from app.rag.query_builder import CategoryQuery
from app.rag.retrieval import HybridRetriever, RetrievedChunk, _rrf_fuse


@pytest.fixture()
def embedder():
    return HashingEmbeddingProvider()


@pytest.fixture()
def vector_store():
    return VectorStoreWriter(client=chromadb.EphemeralClient(), collection_name=f"test-{uuid.uuid4().hex}")


def _ingest(vector_store, embedder, doc_id, title, text, **source_kwargs):
    source = SourceMeta(law_name=source_kwargs.pop("law_name", title), **source_kwargs)
    return ingest_document(doc_id, title, text, source, vector_store=vector_store, embedder=embedder)


def _cq(category: str, query_text: str) -> CategoryQuery:
    return CategoryQuery(category=category, query_text=query_text, matched_traits=[])


class TestRetrievedChunk:
    def test_fused_score_defaults_to_zero(self):
        chunk = RetrievedChunk(chunk_id="x", text="y", metadata={})
        assert chunk.fused_score == 0.0

    def test_bm25_score_defaults_to_zero(self):
        chunk = RetrievedChunk(chunk_id="x", text="y")
        assert chunk.bm25_score == 0.0

    def test_metadata_defaults_to_empty_dict(self):
        chunk = RetrievedChunk(chunk_id="x", text="y")
        assert chunk.metadata == {}


class TestRrfFuse:
    def test_top_of_both_lists_scores_highest(self):
        scores = _rrf_fuse(["a", "b", "c"], ["a", "c", "b"])
        assert scores["a"] > scores["b"]
        assert scores["a"] > scores["c"]

    def test_id_in_only_one_list_still_scores(self):
        scores = _rrf_fuse(["a"], ["b"])
        assert scores["a"] > 0
        assert scores["b"] > 0

    def test_empty_lists_produce_empty_scores(self):
        assert _rrf_fuse([], []) == {}


class TestHybridRetrieverBasics:
    def test_empty_corpus_returns_no_hits(self, vector_store, embedder):
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("tax", "thuế thu nhập cá nhân"), user_entity_type="individual", user_province=None
        )
        assert hits == []

    def test_returns_the_only_matching_chunk(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "law-1",
            "Tax Law",
            "Điều 1. Personal income tax\n1. File an annual return by March 31.\n",
            category="tax",
            entity_type="both",
        )
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("tax", "personal income tax annual filing"),
            user_entity_type="individual",
            user_province=None,
        )
        assert len(hits) == 1
        assert hits[0].metadata["law_name"] == "Tax Law"
        assert hits[0].fused_score > 0

    def test_different_category_is_never_returned(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "law-1",
            "Tax Law",
            "Điều 1. Tax filing.\n",
            category="tax",
            entity_type="both",
        )
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("labor_insurance", "tax filing"), user_entity_type="individual", user_province=None
        )
        assert hits == []

    def test_results_are_sorted_best_first(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "law-1",
            "Tax Filing Law",
            "Điều 1. Personal income tax filing.\n1. File your annual tax return by March 31.\n",
            category="tax",
            entity_type="both",
        )
        _ingest(
            vector_store,
            embedder,
            "law-2",
            "Unrelated Law",
            "Điều 1. Environmental protection.\n1. Do not pollute the river.\n",
            category="tax",  # same category so both are candidates
            entity_type="both",
        )
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("tax", "personal income tax annual filing"),
            user_entity_type="individual",
            user_province=None,
        )
        assert len(hits) == 2
        assert hits[0].fused_score >= hits[1].fused_score
        assert "Tax Filing Law" == hits[0].metadata["law_name"]

    def test_bm25_score_is_carried_and_reflects_lexical_relevance(self, vector_store, embedder):
        # three chunks in the same category; the RRF fused scores can
        # tie (both signals are coarse), so the raw BM25 score is the
        # tiebreaker callers like chat/tools.py rely on — it must be
        # populated and graded, not a constant.
        #
        # NOTE: needs >=3 corpus docs on purpose. rank_bm25's idf is
        # log((N-df+0.5)/(df+0.5)) with no "+1" term, so in a 2-document
        # corpus every term has df=1 -> idf=0 -> every BM25 score is 0
        # (verified empirically while writing this test). Production
        # categories have 20-130 chunks, so the signal is real there;
        # the test just has to avoid the degenerate case.
        _ingest(
            vector_store,
            embedder,
            "law-1",
            "Tax Filing Law",
            "Điều 1. Personal income tax filing.\n1. File your annual tax return by March 31.\n",
            category="tax",
            entity_type="both",
        )
        _ingest(
            vector_store,
            embedder,
            "law-2",
            "Mostly Unrelated Law",
            "Điều 1. Environmental protection and river water quality monitoring reports.\n",
            category="tax",
            entity_type="both",
        )
        _ingest(
            vector_store,
            embedder,
            "law-3",
            "Also Unrelated Law",
            "Điều 1. Fisheries licensing and coastal fishing vessel registration procedures.\n",
            category="tax",
            entity_type="both",
        )
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("tax", "personal income tax filing annual return"),
            user_entity_type="individual",
            user_province=None,
        )
        assert len(hits) == 3
        assert hits[0].bm25_score > hits[1].bm25_score
        assert hits[0].metadata["law_name"] == "Tax Filing Law"


class TestEntityTypeFiltering:
    def test_business_only_chunk_is_hidden_from_individual_user(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "biz-1",
            "Business Registration Law",
            "Điều 1. Register your business within 30 days.\n",
            category="business_licensing",
            entity_type="business",
        )
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("business_licensing", "business registration"),
            user_entity_type="individual",
            user_province=None,
        )
        assert hits == []

    def test_business_only_chunk_is_visible_to_business_user(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "biz-1",
            "Business Registration Law",
            "Điều 1. Register your business within 30 days.\n",
            category="business_licensing",
            entity_type="business",
        )
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("business_licensing", "business registration"),
            user_entity_type="business",
            user_province=None,
        )
        assert len(hits) == 1

    def test_both_entity_type_chunk_is_visible_to_everyone(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "law-1",
            "General Law",
            "Điều 1. Applies to everyone.\n",
            category="tax",
            entity_type="both",
        )
        retriever = HybridRetriever(vector_store, embedder)
        for entity_type in ("individual", "business"):
            hits = retriever.retrieve(
                _cq("tax", "applies to everyone"), user_entity_type=entity_type, user_province=None
            )
            assert len(hits) == 1


class TestProvinceFiltering:
    def test_national_law_is_visible_regardless_of_user_province(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "law-1",
            "National Tax Law",
            "Điều 1. National tax rule.\n",
            category="tax",
            entity_type="both",
        )
        retriever = HybridRetriever(vector_store, embedder)
        for province in (None, "Hanoi", "Phú Thọ"):
            hits = retriever.retrieve(
                _cq("tax", "national tax rule"), user_entity_type="individual", user_province=province
            )
            assert len(hits) == 1

    def test_provincial_law_hidden_without_a_matching_province(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "law-1",
            "Phu Tho Decision",
            "Điều 1. Local registration fee.\n",
            category="residence_civil",
            entity_type="both",
            province_scope="Phú Thọ",
        )
        retriever = HybridRetriever(vector_store, embedder)

        no_province = retriever.retrieve(
            _cq("residence_civil", "local registration fee"),
            user_entity_type="individual",
            user_province=None,
        )
        wrong_province = retriever.retrieve(
            _cq("residence_civil", "local registration fee"),
            user_entity_type="individual",
            user_province="Hanoi",
        )
        assert no_province == []
        assert wrong_province == []

    def test_provincial_law_visible_for_matching_province(self, vector_store, embedder):
        _ingest(
            vector_store,
            embedder,
            "law-1",
            "Phu Tho Decision",
            "Điều 1. Local registration fee.\n",
            category="residence_civil",
            entity_type="both",
            province_scope="Phú Thọ",
        )
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("residence_civil", "local registration fee"),
            user_entity_type="individual",
            user_province="Phú Thọ",
        )
        assert len(hits) == 1


class TestTopNPerSignal:
    def test_top_n_per_signal_bounds_the_result_size(self, vector_store, embedder):
        for i in range(8):
            _ingest(
                vector_store,
                embedder,
                f"law-{i}",
                f"Tax Law {i}",
                f"Điều 1. Tax filing rule number {i}.\n",
                category="tax",
                entity_type="both",
            )
        retriever = HybridRetriever(vector_store, embedder)
        hits = retriever.retrieve(
            _cq("tax", "tax filing rule"),
            user_entity_type="individual",
            user_province=None,
            top_n_per_signal=3,
        )
        # Same 8 candidates rank near-identically on both BM25 and
        # vector signals here (near-identical text), so the fused,
        # de-duplicated result should be bounded near top_n_per_signal,
        # never anywhere close to the full 8-document corpus.
        assert len(hits) <= 6


class _SemanticHashing(HashingEmbeddingProvider):
    """Hashing vectors (deterministic, offline) but flagged as a real
    semantic provider — lets the pruned retrieval path be exercised
    without downloading an embedding model (retrieval tests use this
    deliberately; production uses the local/Gemini/OpenAI providers)."""

    is_semantic = True


class TestPrunedRetrievalPath:
    def test_semantic_embedder_takes_pruned_path(self, vector_store, monkeypatch):
        retriever = HybridRetriever(vector_store, _SemanticHashing())
        calls = {"pruned": 0, "full": 0}

        def spy_pruned(*a, **k):
            calls["pruned"] += 1
            return []

        def spy_full(*a, **k):
            calls["full"] += 1
            return []

        monkeypatch.setattr(retriever, "_retrieve_pruned", spy_pruned)
        monkeypatch.setattr(retriever, "_retrieve_full", spy_full)
        retriever.retrieve(_cq("tax", "x"), user_entity_type="individual", user_province=None)
        assert calls["pruned"] == 1
        assert calls["full"] == 0

    def test_hashing_embedder_takes_full_path(self, vector_store, embedder, monkeypatch):
        retriever = HybridRetriever(vector_store, embedder)
        calls = {"pruned": 0, "full": 0}

        def spy_pruned(*a, **k):
            calls["pruned"] += 1
            return []

        def spy_full(*a, **k):
            calls["full"] += 1
            return []

        monkeypatch.setattr(retriever, "_retrieve_pruned", spy_pruned)
        monkeypatch.setattr(retriever, "_retrieve_full", spy_full)
        retriever.retrieve(_cq("tax", "x"), user_entity_type="individual", user_province=None)
        assert calls["full"] == 1
        assert calls["pruned"] == 0

    def test_pruned_path_returns_same_quality_as_full(self, vector_store):
        sem = _SemanticHashing()
        _ingest(
            vector_store,
            sem,
            "law-1",
            "Tax Filing Law",
            "Điều 1. Personal income tax filing.\n1. File your annual tax return by March 31.\n",
            category="tax",
            entity_type="both",
        )
        _ingest(
            vector_store,
            sem,
            "law-2",
            "Environmental Law",
            "Điều 1. Environmental protection and river water quality monitoring reports.\n",
            category="tax",
            entity_type="both",
        )
        _ingest(
            vector_store,
            sem,
            "law-3",
            "Fisheries Law",
            "Điều 1. Fisheries licensing and coastal fishing vessel registration procedures.\n",
            category="tax",
            entity_type="both",
        )
        retriever = HybridRetriever(vector_store, sem)
        hits = retriever.retrieve(
            _cq("tax", "personal income tax filing annual return"),
            user_entity_type="individual",
            user_province=None,
        )
        # 3 docs fit inside the default pool (100) so the pruned path
        # sees the same candidate set the full path would — and the tax
        # chunk must still win on both signals.
        assert len(hits) == 3
        assert hits[0].metadata["law_name"] == "Tax Filing Law"
        assert hits[0].bm25_score > hits[1].bm25_score

    def test_pruned_path_bounds_the_vector_pool(self, vector_store, monkeypatch):
        sem = _SemanticHashing()
        for i in range(5):
            _ingest(
                vector_store,
                sem,
                f"law-{i}",
                f"Tax Law {i}",
                f"Điều 1. Tax filing rule number {i}.\n",
                category="tax",
                entity_type="both",
            )
        retriever = HybridRetriever(vector_store, sem)
        captured = {}
        original_query = vector_store.query

        def spy(query_embedding, n_results=5, where=None):
            captured["n_results"] = n_results
            return original_query(query_embedding, n_results=n_results, where=where)

        monkeypatch.setattr(vector_store, "query", spy)
        retriever._retrieve_pruned(
            _cq("tax", "tax filing rule"),
            user_entity_type="individual",
            user_province=None,
            top_n_per_signal=2,
            candidate_pool_size=2,
        )
        # n_results is the pool size (2), NOT the full category size (5)
        # — that cap is the entire point of the pruned path.
        assert captured.get("n_results") == 2
