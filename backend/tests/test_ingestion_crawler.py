"""Unit tests for app/ingestion/crawler.py.

Uses a FAKE HTTP client (a plain object satisfying the `HttpClient`
Protocol: `.get(url, params=None, timeout=...)` returning an object with
`.json()`/`.text`/`.raise_for_status()`) against realistic fixture
payloads shaped like the real VBPL gateway API — never the real
network. See crawler.py's module docstring for why the live API
couldn't be exercised from this sandbox, and for the honest scope of
what this test suite does and doesn't prove: it proves the pagination,
status-filtering, HTML-to-text conversion, category classification,
and checkpointing logic are correct against the documented response
shape — the same "prove the real parsing logic without live network"
approach test_news_crawler.py uses for the RSS crawler above.
"""
from __future__ import annotations

import json

import pytest

from app.ingestion.crawler import (
    GenericHtmlDocumentLoader,
    VbplGatewayCrawler,
    _html_to_legal_text,
    classify_category,
)


class _FakeResponse:
    def __init__(self, payload=None, text: str = "", status_ok: bool = True):
        self._payload = payload
        self.text = text
        self._status_ok = status_ok

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("simulated HTTP error")


class _FakeHttpClient:
    """Routes by URL substring to canned responses — enough for a
    crawler that only ever calls two endpoint shapes (list, detail)."""

    def __init__(self, list_pages: dict[int, list[dict]], details: dict[str, dict]):
        self._list_pages = list_pages
        self._details = details
        self.requested_urls: list[str] = []

    def get(self, url, params=None, timeout=30):
        self.requested_urls.append(url)
        if url.endswith("/api/qtdc/public/doc/all"):
            page = (params or {}).get("pageIndex", 0)
            return _FakeResponse(payload=self._list_pages.get(page, []))
        # detail endpoint: .../doc/{id}
        doc_id = url.rsplit("/", 1)[-1]
        detail = self._details.get(doc_id)
        if detail is None:
            return _FakeResponse(payload=None, status_ok=False)
        return _FakeResponse(payload=detail)


def _detail(
    doc_id: str,
    title: str,
    eff_status: str = "Còn hiệu lực",
    content_html: str = "<p>Điều 1. Test</p><p>1. Nội dung.</p>",
) -> dict:
    return {
        "docNum": f"{doc_id}/2026/ND-CP",
        "title": title,
        "effStatus": eff_status,
        "effFrom": "2026-01-01",
        "effTo": None,
        "documentContent": {"content": content_html},
    }


class TestHtmlToLegalText:
    def test_paragraphs_become_separate_lines(self):
        html = "<div><p>Chương I</p><p>Điều 1. Mục đích</p><p>1. Nội dung khoản một.</p></div>"
        text = _html_to_legal_text(html)
        lines = text.splitlines()
        assert "Chương I" in lines
        assert "Điều 1. Mục đích" in lines
        assert "1. Nội dung khoản một." in lines

    def test_br_tags_become_line_breaks(self):
        html = "<p>Điều 1. Test<br>1. First line<br>2. Second line</p>"
        text = _html_to_legal_text(html)
        assert "1. First line" in text.splitlines()
        assert "2. Second line" in text.splitlines()

    def test_blank_lines_collapsed(self):
        html = "<p>Line one</p>\n\n\n<p>Line two</p>"
        text = _html_to_legal_text(html)
        assert "" not in text.splitlines()


class TestClassifyCategory:
    @pytest.mark.parametrize(
        "title,expected",
        [
            ("Luật Thuế thu nhập cá nhân", "tax"),
            ("Nghị định về bảo hiểm xã hội", "labor_insurance"),
            ("Luật Doanh nghiệp", "business_licensing"),
            ("Luật Đất đai", "property_vehicles"),
            ("Luật Hôn nhân và Gia đình", "family_civil"),
        ],
    )
    def test_keyword_match(self, title, expected):
        assert classify_category(title) == expected

    def test_unmatched_title_falls_back_to_default(self):
        assert classify_category("Some Unrelated Title") == "residence_civil"


class TestVbplGatewayCrawler:
    def test_crawls_and_yields_ingestible_documents(self):
        http = _FakeHttpClient(
            list_pages={0: [{"id": "d1"}, {"id": "d2"}]},
            details={
                "d1": _detail("d1", "Luật Thuế thu nhập cá nhân"),
                "d2": _detail("d2", "Luật Doanh nghiệp"),
            },
        )
        crawler = VbplGatewayCrawler(http_client=http, page_size=10, rate_limit_seconds=0)

        docs = list(crawler.crawl(max_documents=10))

        assert len(docs) == 2
        assert docs[0].doc_id == "vbpl-d1"
        assert docs[0].source.category == "tax"
        assert docs[1].source.category == "business_licensing"
        assert "Điều 1" in docs[0].raw_text

    def test_skips_documents_with_non_ingestible_status(self):
        http = _FakeHttpClient(
            list_pages={0: [{"id": "d1"}, {"id": "d2"}]},
            details={
                "d1": _detail("d1", "Old Law", eff_status="Hết hiệu lực toàn bộ"),
                "d2": _detail("d2", "Current Law", eff_status="Còn hiệu lực"),
            },
        )
        crawler = VbplGatewayCrawler(http_client=http, rate_limit_seconds=0)

        docs = list(crawler.crawl(max_documents=10))

        assert len(docs) == 1
        assert docs[0].title == "Current Law"

    def test_skips_documents_with_no_content(self):
        http = _FakeHttpClient(
            list_pages={0: [{"id": "d1"}]},
            details={"d1": _detail("d1", "Empty Law", content_html="")},
        )
        crawler = VbplGatewayCrawler(http_client=http, rate_limit_seconds=0)

        docs = list(crawler.crawl(max_documents=10))
        assert docs == []

    def test_max_documents_caps_yield_even_across_multiple_pages(self):
        http = _FakeHttpClient(
            list_pages={
                0: [{"id": "d1"}, {"id": "d2"}],
                1: [{"id": "d3"}],
            },
            details={
                "d1": _detail("d1", "Law One"),
                "d2": _detail("d2", "Law Two"),
                "d3": _detail("d3", "Law Three"),
            },
        )
        crawler = VbplGatewayCrawler(http_client=http, page_size=2, rate_limit_seconds=0)

        docs = list(crawler.crawl(max_documents=1))
        assert len(docs) == 1

    def test_empty_page_stops_pagination(self):
        http = _FakeHttpClient(list_pages={0: []}, details={})
        crawler = VbplGatewayCrawler(http_client=http, rate_limit_seconds=0)
        assert list(crawler.crawl(max_documents=10)) == []

    def test_one_failed_document_does_not_stop_the_crawl(self):
        # d2 has no detail registered -> _FakeHttpClient returns status_ok=False for it.
        http = _FakeHttpClient(
            list_pages={0: [{"id": "d1"}, {"id": "d2"}]},
            details={"d1": _detail("d1", "Good Law")},
        )
        crawler = VbplGatewayCrawler(http_client=http, rate_limit_seconds=0)

        docs = list(crawler.crawl(max_documents=10))
        assert len(docs) == 1
        assert docs[0].title == "Good Law"

    def test_checkpoint_resumes_from_next_page(self, tmp_path):
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint_path.write_text(json.dumps({"last_page_completed": 0, "failed_doc_ids": []}))

        http = _FakeHttpClient(
            list_pages={
                0: [{"id": "d1"}],  # should NOT be re-fetched — checkpoint says page 0 is done
                1: [{"id": "d2"}],
            },
            details={"d1": _detail("d1", "Already Done"), "d2": _detail("d2", "New Doc")},
        )
        crawler = VbplGatewayCrawler(http_client=http, rate_limit_seconds=0, checkpoint_path=checkpoint_path)

        docs = list(crawler.crawl(max_documents=10))

        assert [d.title for d in docs] == ["New Doc"]

    def test_category_override_takes_precedence_over_classifier(self):
        http = _FakeHttpClient(
            list_pages={0: [{"id": "d1"}]},
            details={"d1": _detail("d1", "Luật Thuế thu nhập cá nhân")},  # would classify as "tax"
        )
        crawler = VbplGatewayCrawler(
            http_client=http, rate_limit_seconds=0, category_overrides={"d1": "family_civil"}
        )

        docs = list(crawler.crawl(max_documents=10))
        assert docs[0].source.category == "family_civil"


class TestGenericHtmlDocumentLoader:
    def test_extracts_content_via_selector(self):
        class _Http:
            def get(self, url, params=None, timeout=30):
                html = (
                    "<html><body><nav>ignore me</nav>"
                    "<article id='main'><p>Điều 1. Test Article</p>"
                    "<p>1. Some content.</p></article></body></html>"
                )
                return _FakeResponse(text=html)

        loader = GenericHtmlDocumentLoader(
            http_client=_Http(),
            documents=[
                {
                    "url": "https://example.com/law",
                    "doc_id": "ext-1",
                    "title": "External Law",
                    "content_selector": "#main",
                    "category": "tax",
                }
            ],
            rate_limit_seconds=0,
        )

        docs = list(loader.crawl())

        assert len(docs) == 1
        assert "ignore me" not in docs[0].raw_text
        assert "Điều 1. Test Article" in docs[0].raw_text
        assert docs[0].source.category == "tax"

    def test_missing_selector_match_is_skipped_not_crashed(self):
        class _Http:
            def get(self, url, params=None, timeout=30):
                return _FakeResponse(text="<html><body><p>no matching element</p></body></html>")

        loader = GenericHtmlDocumentLoader(
            http_client=_Http(),
            documents=[
                {
                    "url": "https://example.com/law",
                    "doc_id": "ext-1",
                    "content_selector": "#does-not-exist",
                }
            ],
            rate_limit_seconds=0,
        )
        assert list(loader.crawl()) == []
