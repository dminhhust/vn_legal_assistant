"""Unit tests for app/ingestion/hf_dataset_loader.py.

Uses fake dataset rows via `rows_provider` injection — never the real
Hugging Face Hub. Row shapes below mirror the real `tmquan/vbpl-vn`
schema (see the module docstring), including the exact failure mode
found by actually running this loader against the real dataset: rows
whose `markdown` is one flowing paragraph with no newlines around
"Điều N" — which produced 0 parsed articles for all 10 real documents
in the first production run, before the normalization fix in this
module. These tests exist specifically to lock that fix in place.
"""
from __future__ import annotations

from app.ingestion.chunker import chunk_document
from app.ingestion.hf_dataset_loader import HfVbplDatasetLoader, _normalize_markdown_for_parser
from app.ingestion.parser import parse_document


def _row(
    doc_name: str = "100",
    title: str = "thành lập Ban Kinh tế Chính phủ",
    markdown: str | None = "Điều 1. Nay thành lập Ban.\nĐiều 2. Có hiệu lực ngay.",
    doc_number: list | None = None,
    legal_type: str = "Sắc lệnh",
    doc_type: str = "sac_lenh",
    legal_area: str = "Chưa phân loại",
    scope: str = "trung_uong",
    issuing_authority: str | None = "Chủ tịch nước",
    issue_date: str = "1950-05-14",
    source_url: str = "https://vbpl.vn/van-ban/chi-tiet/example",
) -> dict:
    return {
        "doc_name": doc_name,
        "title": title,
        "markdown": markdown,
        "doc_number": doc_number if doc_number is not None else ["68/SL"],
        "legal_type": legal_type,
        "doc_type": doc_type,
        "legal_area": legal_area,
        "scope": scope,
        "issuing_authority": issuing_authority,
        "issue_date": issue_date,
        "source_url": source_url,
    }


class TestNormalizeMarkdownForParser:
    def test_inline_dieu_headings_get_line_breaks(self):
        # Reproduces the real, confirmed-broken shape: one flowing
        # paragraph, "Điều N" mentioned inline with no preceding newline.
        raw = (
            "Xét nhu cầu công việc; Điều 1. Nay thành lập Ban Kinh tế Chính phủ. "
            "Điều 2. Ban có nhiệm vụ nghiên cứu kế hoạch. "
            "Điều 3. Có hiệu lực kể từ ngày ký."
        )
        normalized = _normalize_markdown_for_parser(raw)
        lines = [l for l in normalized.splitlines() if l.strip()]
        assert any(l.startswith("Điều 1.") for l in lines)
        assert any(l.startswith("Điều 2.") for l in lines)
        assert any(l.startswith("Điều 3.") for l in lines)

    def test_inline_chuong_headings_get_line_breaks(self):
        raw = "Some preamble text Chương I QUY ĐỊNH CHUNG Điều 1. Nội dung."
        normalized = _normalize_markdown_for_parser(raw)
        lines = [l for l in normalized.splitlines() if l.strip()]
        assert any(l.startswith("Chương I") for l in lines)

    def test_already_linebroken_input_is_not_doubled(self):
        # Already-well-formed input (a real newline already precedes
        # each heading) should pass through without extra blank lines.
        raw = "Preamble.\nĐiều 1. First.\nĐiều 2. Second."
        normalized = _normalize_markdown_for_parser(raw)
        assert normalized == raw

    def test_end_to_end_matches_real_confirmed_failure_shape(self):
        # This exact shape (multi-Điều, single flowing paragraph) is
        # what the real tmquan/vbpl-vn markdown produced in production,
        # and what caused parse_document() to find 0 articles before
        # this normalization existed. Assert the fix actually recovers
        # real articles, not just that the regex runs.
        raw = (
            "SẮC LỆNH CỦA CHỦ TỊCH NƯỚC SỐ 68/SL NGÀY 14 THÁNG 5 NĂM 1950 "
            "Xét nhu cầu công việc; Chương I QUY ĐỊNH CHUNG "
            "Điều 1. Nay thành lập Ban Kinh tế Chính phủ. "
            "Điều 2. Ban Kinh tế Chính phủ có nhiệm vụ nghiên cứu kế hoạch kinh tế. "
            "Điều 3. Sắc lệnh này có hiệu lực kể từ ngày ký."
        )
        normalized = _normalize_markdown_for_parser(raw)
        doc = parse_document("test-100", "thành lập Ban Kinh tế Chính phủ", normalized)

        assert len(doc.chapters) == 1
        assert len(doc.all_articles()) == 3
        numbers = [a.number for a in doc.all_articles()]
        assert numbers == ["1", "2", "3"]

    def test_without_normalization_the_original_bug_reproduces(self):
        # Negative control: confirms this test suite would actually
        # catch a regression — feeding the RAW (unnormalized) text
        # straight to parse_document reproduces the exact 0-articles
        # failure seen in production.
        raw = (
            "Xét nhu cầu công việc; Điều 1. Nay thành lập Ban. "
            "Điều 2. Có hiệu lực ngay."
        )
        doc = parse_document("test-100", "title", raw)
        assert len(doc.all_articles()) == 0

    def test_citation_to_another_law_is_not_treated_as_a_heading(self):
        # "quy định tại Điều 8 của Luật ..." cites ANOTHER document's
        # article — must not become a heading of this one.
        raw = (
            "Điều 1. Phạm vi điều chỉnh. "
            "Điều 2. Việc xử lý vi phạm thực hiện theo quy định tại Điều 15 của Luật Xử lý vi phạm hành chính."
        )
        normalized = _normalize_markdown_for_parser(raw)
        doc = parse_document("test", "title", normalized)
        numbers = [a.number for a in doc.all_articles()]
        assert numbers == ["1", "2"]
        assert "Điều 15" in doc.all_articles()[1].body or "Điều 15" in doc.all_articles()[1].title

    def test_citation_matching_next_real_article_number_does_not_cause_duplicate_or_misattribution(self):
        # The exact production failure: a citation to "Điều 8" of
        # ANOTHER law appears in Điều 7's body, immediately before this
        # document's own real "Điều 8" heading. The citation's number
        # coincides with (in fact equals) the real next article number
        # — this is what defeated the ascending-number check alone and
        # required the citation-preposition check too.
        raw = (
            "Điều 6. Trách nhiệm thi hành. "
            "Điều 7. Việc điều chỉnh dự toán thực hiện theo quy định tại Điều 8 của Luật Ngân sách nhà nước. "
            "Điều 8. Nghị quyết này có hiệu lực kể từ ngày ký."
        )
        normalized = _normalize_markdown_for_parser(raw)
        doc = parse_document("test-100000", "title", normalized)
        numbers = [a.number for a in doc.all_articles()]
        assert numbers == ["6", "7", "8"]  # no duplicate "8", nothing dropped

        article_7 = doc.all_articles()[1]
        article_8 = doc.all_articles()[2]
        assert "của Luật Ngân sách nhà nước" in (article_7.title + article_7.body)
        assert "hiệu lực" in (article_8.title + article_8.body)

        # This is exactly what crashed vector_store.upsert_chunks with
        # chromadb.errors.DuplicateIDError in production.
        chunks = chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_amendment_reference_is_not_treated_as_a_heading(self):
        # "sửa đổi Điều 3 Nghị định số ..." — another common citation
        # phrasing (amending a DIFFERENT document's article).
        raw = (
            "Điều 1. Sửa đổi, bổ sung Điều 3 Nghị định số 15/2020/NĐ-CP. "
            "Điều 2. Hiệu lực thi hành."
        )
        normalized = _normalize_markdown_for_parser(raw)
        doc = parse_document("test", "title", normalized)
        assert [a.number for a in doc.all_articles()] == ["1", "2"]


class TestHfVbplDatasetLoader:
    def test_yields_documents_from_injected_rows(self):
        loader = HfVbplDatasetLoader(rows_provider=lambda: [_row(doc_name="1"), _row(doc_name="2")])
        docs = list(loader.crawl())
        assert len(docs) == 2
        assert docs[0].doc_id == "vbpl-hf-1"
        assert docs[1].doc_id == "vbpl-hf-2"

    def test_null_markdown_rows_are_skipped(self):
        # ~7% of the real dataset has markdown=null (see module
        # docstring) — must not crash or produce an empty ingest.
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(doc_name="1", markdown=None), _row(doc_name="2")]
        )
        docs = list(loader.crawl())
        assert len(docs) == 1
        assert docs[0].doc_id == "vbpl-hf-2"

    def test_blank_markdown_rows_are_skipped(self):
        loader = HfVbplDatasetLoader(rows_provider=lambda: [_row(doc_name="1", markdown="   ")])
        assert list(loader.crawl()) == []

    def test_yielded_document_is_actually_parseable_into_articles(self):
        # End-to-end: the CrawledDocument this loader produces must be
        # directly usable by app.ingestion.parser, not just non-empty.
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [
                _row(markdown="Điều 1. Nay thành lập Ban. Điều 2. Có hiệu lực ngay.")
            ]
        )
        [doc] = list(loader.crawl())
        parsed = parse_document(doc.doc_id, doc.title, doc.raw_text)
        assert len(parsed.all_articles()) == 2

    def test_max_documents_caps_yield(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(doc_name=str(i)) for i in range(5)]
        )
        docs = list(loader.crawl(max_documents=2))
        assert len(docs) == 2

    def test_category_classified_from_title(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(title="Luật Thuế thu nhập cá nhân")]
        )
        [doc] = list(loader.crawl())
        assert doc.source.category == "tax"

    def test_category_override_takes_precedence(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(doc_name="1", title="Luật Thuế thu nhập cá nhân")],
            category_overrides={"1": "family_civil"},
        )
        [doc] = list(loader.crawl())
        assert doc.source.category == "family_civil"

    def test_law_name_includes_doc_number_and_type(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(legal_type="Sắc lệnh", doc_number=["68/SL"], title="thành lập Ban")]
        )
        [doc] = list(loader.crawl())
        assert "Sắc lệnh" in doc.source.law_name
        assert "68/SL" in doc.source.law_name
        assert "thành lập Ban" in doc.source.law_name

    def test_missing_doc_number_falls_back_gracefully(self):
        loader = HfVbplDatasetLoader(rows_provider=lambda: [_row(doc_number=[])])
        [doc] = list(loader.crawl())
        assert doc.source.law_name  # non-empty, no crash

    def test_source_url_and_effective_from_carried_through(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(issue_date="1950-05-14", source_url="https://vbpl.vn/x")]
        )
        [doc] = list(loader.crawl())
        assert doc.source.effective_from == "1950-05-14"
        assert doc.source.source_url == "https://vbpl.vn/x"


class TestSchemaAlignment:
    """Covers the fields present in the real dataset schema (confirmed
    against the tmquan/vbpl-vn dataset card) that the first version of
    this loader either ignored or handled only approximately."""

    def test_translation_doc_type_excluded_by_default(self):
        # ban_dich_van_ban = a TRANSLATION of another document, not a
        # binding legal instrument itself (6.7% of the real dataset).
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(doc_name="1", doc_type="ban_dich_van_ban"), _row(doc_name="2")]
        )
        docs = list(loader.crawl())
        assert [d.doc_id for d in docs] == ["vbpl-hf-2"]

    def test_official_correspondence_doc_type_excluded_by_default(self):
        # cong_van = official dispatch/correspondence, not a "văn bản
        # quy phạm pháp luật" under the law this taxonomy follows.
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(doc_name="1", doc_type="cong_van"), _row(doc_name="2")]
        )
        docs = list(loader.crawl())
        assert [d.doc_id for d in docs] == ["vbpl-hf-2"]

    def test_excluded_doc_types_is_overridable(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(doc_name="1", doc_type="ban_dich_van_ban")],
            excluded_doc_types=frozenset(),  # opt back in
        )
        docs = list(loader.crawl())
        assert [d.doc_id for d in docs] == ["vbpl-hf-1"]

    def test_normative_doc_types_are_not_excluded(self):
        for doc_type in [
            "luat",
            "nghi_dinh",
            "thong_tu",
            "quyet_dinh",
            "nghi_quyet",
            "chi_thi",
            "sac_lenh",
            # Confirmed real, if less common, normative slugs (see
            # DEFAULT_EXCLUDED_DOC_TYPES's comment for the source) —
            # must not be caught by the wider exclusion list either.
            "sac_luat",
            "nghi_quyet_lien_tich",
            "thong_tu_lien_bo",
            "van_ban_hop_nhat",
        ]:
            loader = HfVbplDatasetLoader(rows_provider=lambda dt=doc_type: [_row(doc_type=dt)])
            assert len(list(loader.crawl())) == 1, f"{doc_type} should not be excluded"

    def test_non_normative_auxiliary_doc_types_are_excluded_by_default(self):
        # Confirmed via the full doc_type slug enumeration (codes.py,
        # see DEFAULT_EXCLUDED_DOC_TYPES's comment) — none of these are
        # "văn bản quy phạm pháp luật", just auxiliary/diplomatic
        # material that shouldn't be ingested as if it were primary
        # legal text.
        for doc_type in [
            "thong_bao",
            "hiep_dinh",
            "nghi_dinh_thu",
            "ban_ghi_nho",
            "thoa_thuan",
            "van_ban_hanh_chinh_lien_quan",
            "van_ban_khac",
            "van_ban_lien_quan",
            "chuong_trinh",
            "chua_xac_dinh",
        ]:
            loader = HfVbplDatasetLoader(rows_provider=lambda dt=doc_type: [_row(doc_type=dt)])
            assert list(loader.crawl()) == [], f"{doc_type} should be excluded by default"

    def test_multiple_doc_numbers_are_all_kept(self):
        # doc_number is a LIST — some rows carry more than one
        # identifier. The first version of this loader dropped all but
        # doc_number[0].
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(doc_number=["142/2009/QĐ-TTg", "49/2012/QĐ-TTg"])]
        )
        [doc] = list(loader.crawl())
        assert "142/2009/QĐ-TTg" in doc.source.law_name
        assert "49/2012/QĐ-TTg" in doc.source.law_name

    def test_populated_legal_area_improves_classification(self):
        # Title alone ("Ban hành quy định") gives no keyword match and
        # falls back to the default bucket; legal_area supplies "thuế".
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [
                _row(title="Ban hành quy định", legal_area="Quản lý thuế, phí và lệ phí")
            ]
        )
        [doc] = list(loader.crawl())
        assert doc.source.category == "tax"

    def test_unclassified_legal_area_falls_back_to_title(self):
        # "Chưa phân loại" (uncategorised) is the real dataset's
        # literal sentinel for ~71% of rows — must not be treated as a
        # real subject-area signal (e.g. matched as a keyword itself).
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [
                _row(title="Luật Thuế thu nhập cá nhân", legal_area="Chưa phân loại")
            ]
        )
        [doc] = list(loader.crawl())
        assert doc.source.category == "tax"

    def test_province_extracted_for_provincial_scope(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [
                _row(scope="dia_phuong", issuing_authority="UBND tỉnh Bà Rịa - Vũng Tàu")
            ]
        )
        [doc] = list(loader.crawl())
        assert doc.source.province_scope == "Bà Rịa - Vũng Tàu"

    def test_province_extracted_for_city_authority(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [
                _row(scope="dia_phuong", issuing_authority="UBND Thành phố Hồ Chí Minh")
            ]
        )
        [doc] = list(loader.crawl())
        assert doc.source.province_scope == "Hồ Chí Minh"

    def test_central_scope_has_no_province(self):
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(scope="trung_uong", issuing_authority="Bộ Tài chính")]
        )
        [doc] = list(loader.crawl())
        assert doc.source.province_scope is None

    def test_unrecognized_provincial_authority_leaves_province_unset(self):
        # Best-effort: an issuing_authority that doesn't follow the
        # "tỉnh X" / "thành phố X" pattern must not crash or guess.
        loader = HfVbplDatasetLoader(
            rows_provider=lambda: [_row(scope="dia_phuong", issuing_authority="Sở Tư pháp")]
        )
        [doc] = list(loader.crawl())
        assert doc.source.province_scope is None
