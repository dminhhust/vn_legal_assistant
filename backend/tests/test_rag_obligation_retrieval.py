"""Offline, fixture-driven test of the full retrieval pipeline —
same convention as tests/test_ingestion_hf_dataset_loader.py: no live
DB, vector server, or network, a small synthetic corpus standing in
for the corpus properties the caller's brief called out:

  - a national law with two articles                              -> tests hierarchy + article citation
  - a provincial decision, matched only for its own province        -> tests the two-facet jurisdiction filter
  - a document replaced by a newer one via statute_refs             -> tests the consolidation walk
  - a document repealed with nothing shown replacing it             -> tests unresolved-consolidation flagging
  - a metadata-only document (markdown=None) matching jurisdiction  -> tests the null-markdown gap pass
"""
from __future__ import annotations

import math

from app.rag.embeddings import normalize_for_embedding
from app.rag.obligation_retrieval import ObligationRetriever
from app.rag.schema import JurisdictionFacets, LegalDocument

# --- fixture corpus ---------------------------------------------------------

LAW_INCOME_TAX = LegalDocument(
    doc_name="luat-thue-tncn-2007",
    title="Luật Thuế thu nhập cá nhân",
    doc_type="luat",
    legal_type="Luật",
    scope="trung_uong",
    issuing_authority="Quốc hội",
    legal_area="Quản lý thuế, phí và lệ phí",
    issue_date="2007-11-21",
    markdown=(
        "Điều 1. Phạm vi điều chỉnh. Luật này quy định về thuế thu nhập cá nhân. "
        "Điều 2. Đối tượng nộp thuế. Cá nhân cư trú có thu nhập chịu thuế phải nộp thuế thu nhập cá nhân."
    ),
    # Real shape: a flat top-level "sections" list (see
    # app/rag/obligation_retrieval.py's confirmed-shape note), not a nested
    # "children" tree — these fixtures previously used the wrong shape
    # and would have found zero article nodes against the real code.
    structure_json={
        "sections": [
            {"kind": "dieu", "label": "Điều 1", "start": 0, "end": 75},
            {"kind": "dieu", "label": "Điều 2", "start": 76, "end": 176},
        ]
    },
)

DECISION_PHU_THO = LegalDocument(
    doc_name="qd-phu-tho-2022-001",
    title="Quyết định về mức thu phí đăng ký cư trú",
    doc_type="quyet_dinh",
    legal_type="Quyết định",
    scope="dia_phuong",
    issuing_authority="UBND Tỉnh Phú Thọ",
    legal_area=None,  # "Chưa phân loại" case
    issue_date="2022-03-01",
    markdown="Điều 1. Mức thu phí đăng ký cư trú trên địa bàn tỉnh là 50.000 đồng.",
    structure_json={"sections": [{"kind": "dieu", "label": "Điều 1", "start": 0, "end": 70}]},
)

DECREE_OLD = LegalDocument(
    doc_name="nd-2015-cu",
    title="Nghị định cũ về đăng ký kinh doanh",
    doc_type="nghi_dinh",
    legal_type="Nghị định",
    scope="trung_uong",
    issuing_authority="Chính phủ",
    legal_area="Đăng ký kinh doanh",
    issue_date="2015-01-01",
    markdown="Điều 1. Quy định về đăng ký kinh doanh hộ cá thể.",
    structure_json={"sections": [{"kind": "dieu", "label": "Điều 1", "start": 0, "end": 50}]},
    extracted_json={
        "statute_refs": [
            {"target_doc_name": "nd-2020-moi", "relation": "replace", "confidence": 0.9}
        ]
    },
)

DECREE_NEW = LegalDocument(
    doc_name="nd-2020-moi",
    title="Nghị định mới về đăng ký kinh doanh",
    doc_type="nghi_dinh",
    legal_type="Nghị định",
    scope="trung_uong",
    issuing_authority="Chính phủ",
    legal_area="Đăng ký kinh doanh",
    issue_date="2020-06-01",
    markdown="Điều 1. Quy định về đăng ký kinh doanh hộ cá thể (thay thế Nghị định cũ).",
    structure_json={"sections": [{"kind": "dieu", "label": "Điều 1", "start": 0, "end": 70}]},
)

CIRCULAR_REPEALED = LegalDocument(
    doc_name="tt-2010-bai-bo",
    title="Thông tư đã bị bãi bỏ về lệ phí",
    doc_type="thong_tu",
    legal_type="Thông tư",
    scope="trung_uong",
    issuing_authority="Bộ Tài chính",
    legal_area="Quản lý thuế, phí và lệ phí",
    issue_date="2010-01-01",
    markdown="Điều 1. Quy định về lệ phí cũ.",
    structure_json={"sections": [{"kind": "dieu", "label": "Điều 1", "start": 0, "end": 30}]},
    extracted_json={"statute_refs": [{"target_doc_name": None, "relation": "repeal"}]},
)

METADATA_ONLY = LegalDocument(
    doc_name="qd-2011-shell",
    title="Quyết định về thuế thu nhập cá nhân (văn bản cũ, không còn nội dung)",
    doc_type="quyet_dinh",
    legal_type="Quyết định",
    scope="trung_uong",
    issuing_authority="Bộ Tài chính",
    legal_area=None,
    issue_date="2011-01-01",
    markdown=None,  # metadata-only row, body_source == "shell_html"
)

ALL_DOCS = [
    LAW_INCOME_TAX,
    DECISION_PHU_THO,
    DECREE_OLD,
    DECREE_NEW,
    CIRCULAR_REPEALED,
    METADATA_ONLY,
]


# --- fixture embedder + store (no real model / DB) --------------------------


def _bag_of_words_vector(text: str, vocab: list[str]) -> list[float]:
    normalized = normalize_for_embedding(text).casefold()
    return [float(normalized.count(term.casefold())) for term in vocab]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


class FakeEmbedder:
    """Deterministic bag-of-words stand-in for nvidia/llama-nemotron-
    embed-1b-v2 — exercises the pipeline's embedding call shape
    without needing real inference."""

    def __init__(self, vocab: list[str]):
        self.vocab = vocab

    def embed_documents(self, texts):
        return [_bag_of_words_vector(t, self.vocab) for t in texts]

    def embed_query(self, text):
        return _bag_of_words_vector(text, self.vocab)


_VOCAB = ["thuế", "thu nhập", "cá nhân", "đăng ký", "kinh doanh", "phí", "cư trú"]


class InMemoryDocStore:
    def __init__(self, docs: list[LegalDocument], embedder: FakeEmbedder):
        self._docs = {d.doc_name: d for d in docs}
        self._embedder = embedder
        self._doc_vectors = {
            d.doc_name: _bag_of_words_vector(d.markdown, embedder.vocab)
            for d in docs
            if d.has_body
        }

    def get(self, doc_name):
        return self._docs.get(doc_name)

    def iter_by_jurisdiction(self, facets: JurisdictionFacets):
        from app.rag.jurisdiction import matches_jurisdiction

        return [d for d in self._docs.values() if matches_jurisdiction(d, facets)]

    def search_fulltext(self, query, facets, limit):
        from app.rag.jurisdiction import matches_jurisdiction

        q_terms = [t for t in normalize_for_embedding(query).casefold().split() if t]
        hits = []
        for doc in self._docs.values():
            if not doc.has_body or not matches_jurisdiction(doc, facets):
                continue
            body = normalize_for_embedding(doc.markdown).casefold()
            overlap = sum(1 for t in q_terms if t in body)
            if overlap:
                hits.append((doc, min(1.0, overlap / max(1, len(q_terms)))))
        hits.sort(key=lambda pair: -pair[1])
        return hits[:limit]

    def search_embeddings(self, query_vector, facets, limit):
        from app.rag.jurisdiction import matches_jurisdiction

        hits = []
        for name, vec in self._doc_vectors.items():
            doc = self._docs[name]
            if not matches_jurisdiction(doc, facets):
                continue
            hits.append((doc, _cosine(query_vector, vec)))
        hits.sort(key=lambda pair: -pair[1])
        return hits[:limit]


def _make_retriever():
    embedder = FakeEmbedder(_VOCAB)
    store = InMemoryDocStore(ALL_DOCS, embedder)
    return ObligationRetriever(store=store, embedder=embedder)


# --- tests -------------------------------------------------------------------


def test_national_law_is_always_in_scope_without_a_province():
    retriever = _make_retriever()
    result = retriever.retrieve("thuế thu nhập cá nhân", user_province=None)
    doc_names = {hit.citation.doc_name for hit in result.obligations}
    assert "luat-thue-tncn-2007" in doc_names


def test_provincial_document_only_surfaces_for_its_own_province():
    retriever = _make_retriever()

    other_province = retriever.retrieve("phí đăng ký cư trú", user_province="Hà Nam")
    assert "qd-phu-tho-2022-001" not in {h.citation.doc_name for h in other_province.obligations}

    right_province = retriever.retrieve("phí đăng ký cư trú", user_province="Phú Thọ")
    assert "qd-phu-tho-2022-001" in {h.citation.doc_name for h in right_province.obligations}


def test_hierarchy_ranks_luat_above_quyet_dinh():
    retriever = _make_retriever()
    result = retriever.retrieve("thuế thu nhập cá nhân", user_province="Phú Thọ")
    ranks = {hit.citation.doc_name: hit.hierarchy_rank for hit in result.obligations}
    assert ranks["luat-thue-tncn-2007"] < ranks.get("qd-phu-tho-2022-001", 999)


def test_article_level_citation_not_whole_document():
    retriever = _make_retriever()
    result = retriever.retrieve("thu nhập cá nhân", user_province=None)
    law_hit = next(h for h in result.obligations if h.citation.doc_name == "luat-thue-tncn-2007")
    assert law_hit.citation.dieu_number in (1, 2)
    assert law_hit.citation.excerpt  # a real excerpt, not empty


def test_consolidation_walk_prefers_the_replacing_document():
    retriever = _make_retriever()
    result = retriever.retrieve("đăng ký kinh doanh", user_province=None)
    cited_names = {hit.citation.doc_name for hit in result.obligations}
    # the old decree should never be cited as-is; the walk should have
    # resolved it to the replacing decree instead
    assert "nd-2015-cu" not in cited_names
    assert "nd-2020-moi" in cited_names


def test_unresolved_repeal_is_flagged_not_silently_dropped():
    retriever = _make_retriever()
    result = retriever.retrieve("phí", user_province=None)
    unresolved = [g for g in result.gaps if g.kind == "unresolved_consolidation"]
    assert any(g.doc_name == "tt-2010-bai-bo" for g in unresolved)


def test_metadata_only_document_is_flagged_as_a_gap_not_invisible():
    retriever = _make_retriever()
    result = retriever.retrieve("thuế", user_province=None)
    null_md_gaps = [g for g in result.gaps if g.kind == "null_markdown_match"]
    assert any(g.doc_name == "qd-2011-shell" for g in null_md_gaps)
    # and critically: it must never appear as a trusted obligation hit,
    # since there is no body text to have actually matched anything
    assert "qd-2011-shell" not in {h.citation.doc_name for h in result.obligations}


def test_legal_area_is_a_boost_not_a_filter():
    """A document whose legal_area is untagged ("Chưa phân loại" /
    None) must still be retrievable by lexical/embedding match alone —
    filtering on legal_area would silently drop it."""
    retriever = _make_retriever()
    result = retriever.retrieve("phí đăng ký cư trú", user_province="Phú Thọ")
    assert "qd-phu-tho-2022-001" in {h.citation.doc_name for h in result.obligations}
