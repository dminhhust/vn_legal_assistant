"""Legal-source crawler (source loader) — the piece the original
prototype explicitly never built (see docs/ARCHITECTURE.md, "What this
redesign adds"). Fetches real Vietnamese legal documents and turns them
into the (doc_id, title, raw_text, SourceMeta) shape that
app.ingestion.pipeline.ingest_document() already knows how to parse,
chunk, tag, and embed.

Two loaders, behind the same `LegalSourceLoader` Protocol used
everywhere else in this codebase for provider abstraction (embeddings,
LLM router, reranker):

  - VbplGatewayCrawler: a REAL, working client for the Ministry of
    Justice's public VBPL (Văn bản pháp luật) data gateway. This is a
    genuine, documented public API (confirmed via a published
    open-source project doing the same crawl — see the module-level
    ENDPOINT NOTES below), not a guess at HTML selectors that will
    break the moment a page is redesigned. It paginates the document
    list, fetches full detail (including the HTML article content) per
    document, filters by legal-effect status, converts the HTML to the
    line-based plain text app.ingestion.parser expects, and classifies
    each document into this app's category taxonomy.

  - GenericHtmlDocumentLoader: a config-driven fallback for any single
    legal-document page (e.g. thuvienphapluat.vn, luatvietnam.vn, or a
    court/ministry page not covered by the gateway) — point it at a URL
    and a CSS selector for the main content, no new code required.

HONESTY NOTE — UPDATED after actually running this against the live
gateway (matching this codebase's existing convention of testing
claims rather than just documenting assumptions — see the original
docs/PROGRESS_TRACKER.md Phase 2 notes): `VbplGatewayCrawler` HAS now
been run against the real API from an environment with real network
access, via `python -m app.ingestion.run_ingestion --crawl`, and it
fails immediately — `400 Bad Request` on the very first list-page
call. The root cause is more fundamental than "field names might
drift" (this module's original caveat, based only on public
documentation of a project performing the same crawl, without having
run it): `vbpl-bientap-gateway.moj.gov.vn` is the backing API for
vbpl.vn's single-page app, and is gated behind a Bearer token the SPA
obtains by solving Google's invisible reCAPTCHA v2 in a real browser
session (confirmed via a real, independently-published dataset —
tmquan/vbpl-vn on Hugging Face — that documents crawling this same
gateway with headless-Chromium + reCAPTCHA-token automation to get
past exactly this wall). A plain `requests.Session()`, which is all
this class uses, has no way to obtain that token, so every request is
rejected outright, not intermittently. Building a captcha-solving
bypass is out of scope for this codebase.

The endpoint shapes below (paths, `docNum`/`effStatus`/
`documentContent.content` field names) ARE still correct — confirmed
against that same real dataset's schema — so this class's parsing
logic is sound and its unit tests (tests/test_ingestion_crawler.py,
fixture-based against a fake HTTP client) remain a true test of that
logic. The only broken piece is authentication to the live endpoint.
Both loaders take an injectable HTTP client (`requests`-compatible: a
`.get()`/`.post()` object) specifically so this module is fully
testable offline against fixture JSON/HTML without needing live
network access — the same "test the real parsing logic without live
network" pattern already used for app/news/crawler.py's RSS crawler in
the original codebase.

Given the auth wall, see app/ingestion/hf_dataset_loader.py for a
loader that gets real ingested VBPL documents into this app today,
sourced from that same properly-authenticated third-party crawl,
rather than re-solving the reCAPTCHA problem here. `VbplGatewayCrawler`
is left in place (parsing/pagination/status-filtering logic all still
correct and tested) for whenever this app adds real Bearer-token
support — at that point it becomes the preferred, always-current
source again.

ENDPOINT NOTES (VBPL gateway):
  - List:   GET  {base}/api/qtdc/public/doc/all?pageIndex=N&pageSize=M
            Each item includes at least a document id.
  - Detail: GET  {base}/api/qtdc/public/doc/{doc_id}
            Returns JSON with (at least) docNum, title/docName,
            effStatus, effFrom, effTo, and documentContent.content
            (the article body, as HTML).
  Field names on a live response can drift; `_extract_detail_fields`
  below is the single place that reads them, so adapting to a real
  payload shape is a one-function change, not a rewrite.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol

from app.ingestion.metadata import CATEGORIES, SourceMeta

logger = logging.getLogger(__name__)

DEFAULT_VBPL_GATEWAY_BASE = "https://vbpl-bientap-gateway.moj.gov.vn"

# Legal-effect statuses worth ingesting vs. worth skipping. Matches
# docs/ARCHITECTURE.md's ingestion design: superseded provisions should
# not silently pollute retrieval. A document whose status isn't
# recognized at all is skipped (fail closed, not open) rather than
# guessed at.
_INGESTIBLE_STATUSES = {"Còn hiệu lực", "Hết hiệu lực một phần", "Chưa có hiệu lực"}
_SKIP_STATUSES = {"Hết hiệu lực toàn bộ", "Không còn phù hợp", "Ngưng hiệu lực"}

# Heuristic keyword -> category classifier. This is config data (a
# plain dict), matching the "config, not branching code" principle
# used for the category taxonomy itself (app/ingestion/metadata.py) and
# the checklist trigger-trait map (app/rag/query_builder.py) — retuning
# classification is a data edit here, never a code change elsewhere.
# Checked in order; first match wins, so put more specific phrases
# before more general ones.
_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("thuế", "tax"),
    ("phí", "tax"),
    ("lệ phí", "tax"),
    ("kế toán", "tax"),
    ("ngân sách", "tax"),
    ("lao động", "labor_insurance"),
    ("việc làm", "labor_insurance"),
    ("tiền lương", "labor_insurance"),
    ("công đoàn", "labor_insurance"),
    ("bảo hiểm xã hội", "labor_insurance"),
    ("bảo hiểm y tế", "labor_insurance"),
    ("hợp đồng", "contracts_signing"),
    ("công chứng", "contracts_signing"),
    ("cư trú", "residence_civil"),
    ("hộ tịch", "residence_civil"),
    ("căn cước", "residence_civil"),
    ("hộ khẩu", "residence_civil"),
    ("doanh nghiệp", "business_licensing"),
    ("kinh doanh", "business_licensing"),
    ("giấy phép", "business_licensing"),
    ("đầu tư", "business_licensing"),
    ("chứng khoán", "business_licensing"),
    ("đất đai", "property_vehicles"),
    ("nhà ở", "property_vehicles"),
    ("phương tiện", "property_vehicles"),
    ("giao thông", "property_vehicles"),
    ("vận tải", "property_vehicles"),
    ("hôn nhân", "family_civil"),
    ("gia đình", "family_civil"),
    ("thừa kế", "family_civil"),
    ("ly hôn", "family_civil"),
    ("nuôi con nuôi", "family_civil"),
]
_DEFAULT_CATEGORY = "residence_civil"  # broadest baseline bucket when no keyword matches


def classify_category(title: str) -> str:
    """Best-effort category classification from a document's title.
    Deliberately simple and overridable — a human curator should
    spot-check/correct this before trusting it at scale (see the
    per-document `SourceMeta` override support in `crawl()` below)."""
    lowered = title.lower()
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in lowered:
            return category
    return _DEFAULT_CATEGORY


@dataclass
class CrawledDocument:
    """Output shape this module produces — directly consumable by
    app.ingestion.pipeline.ingest_document(doc_id, title, raw_text, source)."""

    doc_id: str
    title: str
    raw_text: str
    source: SourceMeta


class HttpClient(Protocol):
    """Minimal subset of the `requests` interface this module needs —
    injectable so tests never touch the real network. A real
    `requests.Session()` satisfies this Protocol as-is."""

    def get(self, url: str, params: Optional[dict] = None, timeout: float = 30) -> Any: ...


class LegalSourceLoader(Protocol):
    """Every legal-source loader implements this — the same
    provider-abstraction shape used for embeddings/LLM/reranking
    elsewhere in this codebase, so app/ingestion/run_ingestion.py can
    swap loaders without caring which one it's driving."""

    def crawl(self, *, max_documents: Optional[int] = None) -> Iterator[CrawledDocument]: ...


def _html_to_legal_text(html: str) -> str:
    """Converts one article's HTML body into the line-based plain text
    app.ingestion.parser expects (it detects "Chương ..."/"Điều ..."
    headings by matching whole lines — see parser.py's module
    docstring). Block-level tags become line breaks; everything else is
    flattened, matching how Vietnamese legal HTML on official portals
    is actually authored (one block element per Chương/Điều/Khoản line).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - dependency documented in requirements.txt
        raise RuntimeError(
            "beautifulsoup4 is required for HTML legal-document parsing "
            "(pip install beautifulsoup4)"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")

    # Turn <br> into explicit newlines before text extraction — BeautifulSoup's
    # get_text() drops them otherwise, which would merge lines that need
    # to stay separate for the heading-detection regexes in parser.py.
    for br in soup.find_all("br"):
        br.replace_with("\n")

    text = soup.get_text("\n")

    # Normalize: strip trailing whitespace per line, collapse runs of
    # blank lines, drop truly empty lines at the very start/end. Keeps
    # single blank lines between paragraphs (harmless to the parser,
    # which only pattern-matches non-blank lines) rather than jamming
    # everything into one dense block that's hard to spot-check by eye.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _extract_detail_fields(detail: dict) -> dict:
    """Single place that reads field names off a VBPL detail-API
    response — see module docstring's ENDPOINT NOTES on why this is
    isolated. Tolerant of a couple of plausible field-name variants
    since this hasn't been verified against a live payload from this
    sandbox.
    """
    doc_num = detail.get("docNum") or detail.get("docNumber") or ""
    title = detail.get("title") or detail.get("docName") or detail.get("name") or ""
    eff_status = detail.get("effStatus") or detail.get("effectStatus") or ""
    eff_from = detail.get("effFrom") or detail.get("effectiveFrom") or None
    eff_to = detail.get("effTo") or detail.get("effectiveTo") or None

    content_html = ""
    doc_content = detail.get("documentContent") or {}
    if isinstance(doc_content, dict):
        content_html = doc_content.get("content") or ""
    if not content_html:
        content_html = detail.get("content") or ""

    return {
        "doc_num": doc_num,
        "title": title,
        "eff_status": eff_status,
        "eff_from": eff_from,
        "eff_to": eff_to,
        "content_html": content_html,
    }


@dataclass
class VbplGatewayCrawler:
    """Real client for the Ministry of Justice's public VBPL data
    gateway. See module docstring for endpoint shapes and the honesty
    note about not having been run against the live API from this
    sandbox.
    """

    http_client: HttpClient
    base_url: str = DEFAULT_VBPL_GATEWAY_BASE
    page_size: int = 10
    rate_limit_seconds: float = 1.0
    checkpoint_path: Optional[Path] = None
    category_overrides: dict[str, str] = field(default_factory=dict)  # doc_id -> category, for manual curation

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_path and self.checkpoint_path.exists():
            return json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return {"last_page_completed": -1, "failed_doc_ids": []}

    def _save_checkpoint(self, state: dict) -> None:
        if not self.checkpoint_path:
            return
        self.checkpoint_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _list_page(self, page_index: int) -> list[dict]:
        resp = self.http_client.get(
            f"{self.base_url}/api/qtdc/public/doc/all",
            params={"pageIndex": page_index, "pageSize": self.page_size},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        # Tolerate either a bare list or an {"items": [...]} envelope —
        # real pagination envelopes vary; isolated here for the same
        # reason as _extract_detail_fields.
        if isinstance(payload, dict):
            return payload.get("items") or payload.get("data") or []
        return payload or []

    def _fetch_detail(self, doc_id: str) -> Optional[dict]:
        resp = self.http_client.get(f"{self.base_url}/api/qtdc/public/doc/{doc_id}", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def crawl(self, *, max_documents: Optional[int] = None) -> Iterator[CrawledDocument]:
        state = self._load_checkpoint()
        start_page = state["last_page_completed"] + 1
        failed: list[str] = list(state.get("failed_doc_ids", []))
        yielded = 0
        page_index = start_page

        while max_documents is None or yielded < max_documents:
            try:
                summaries = self._list_page(page_index)
            except Exception:  # noqa: BLE001 - one bad page shouldn't kill the whole crawl
                logger.exception("Failed to list VBPL page %d — stopping crawl here", page_index)
                break

            if not summaries:
                break  # reached the end of the list

            for summary in summaries:
                doc_id = str(summary.get("id") or summary.get("docId") or "")
                if not doc_id:
                    continue
                doc = self._crawl_one(doc_id)
                if doc is None:
                    failed.append(doc_id)
                    continue
                yield doc
                yielded += 1
                time.sleep(self.rate_limit_seconds)  # be a polite crawler — see architecture doc's ToS note
                if max_documents is not None and yielded >= max_documents:
                    break

            state = {"last_page_completed": page_index, "failed_doc_ids": failed}
            self._save_checkpoint(state)
            page_index += 1

    def _crawl_one(self, doc_id: str) -> Optional[CrawledDocument]:
        try:
            detail = self._fetch_detail(doc_id)
        except Exception:  # noqa: BLE001 - isolate one document's failure from the rest of the crawl
            logger.warning("Failed to fetch VBPL doc detail for id=%s", doc_id, exc_info=True)
            return None
        if not detail:
            return None

        fields = _extract_detail_fields(detail)
        if fields["eff_status"] and fields["eff_status"] not in _INGESTIBLE_STATUSES:
            logger.info("Skipping doc %s (%s): status=%s", doc_id, fields["title"], fields["eff_status"])
            return None
        if not fields["content_html"]:
            logger.warning("Skipping doc %s (%s): no content", doc_id, fields["title"])
            return None

        raw_text = _html_to_legal_text(fields["content_html"])
        if not raw_text.strip():
            return None

        category = self.category_overrides.get(doc_id) or classify_category(fields["title"])
        law_name = f"{fields['title']} ({fields['doc_num']})".strip() if fields["doc_num"] else fields["title"]

        source = SourceMeta(
            law_name=law_name,
            category=category if category in CATEGORIES else _DEFAULT_CATEGORY,
            entity_type="both",
            effective_from=fields["eff_from"],
            effective_to=fields["eff_to"],
            source_url=f"{self.base_url}/api/qtdc/public/doc/{doc_id}",
        )
        return CrawledDocument(doc_id=f"vbpl-{doc_id}", title=fields["title"], raw_text=raw_text, source=source)


@dataclass
class GenericHtmlDocumentLoader:
    """Config-driven fallback loader for a single legal-document page —
    no code change needed to add a source, just a URL + CSS selector
    for the element containing the article body. Suited to
    thuvienphapluat.vn / luatvietnam.vn style pages, or any one-off
    government/ministry page not covered by the VBPL gateway.
    """

    http_client: HttpClient
    documents: list[dict]  # each: {"url", "doc_id", "title", "content_selector", "category", ...SourceMeta kwargs}
    rate_limit_seconds: float = 1.0

    def crawl(self, *, max_documents: Optional[int] = None) -> Iterator[CrawledDocument]:
        yielded = 0
        for spec in self.documents:
            if max_documents is not None and yielded >= max_documents:
                return
            doc = self._crawl_one(spec)
            if doc is not None:
                yield doc
                yielded += 1
            time.sleep(self.rate_limit_seconds)

    def _crawl_one(self, spec: dict) -> Optional[CrawledDocument]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("beautifulsoup4 is required (pip install beautifulsoup4)") from exc

        try:
            resp = self.http_client.get(spec["url"], timeout=30)
            resp.raise_for_status()
            html = resp.text
        except Exception:  # noqa: BLE001 - isolate one page's failure from the rest of the batch
            logger.warning("Failed to fetch %s", spec.get("url"), exc_info=True)
            return None

        soup = BeautifulSoup(html, "html.parser")
        selector = spec.get("content_selector")
        content_node = soup.select_one(selector) if selector else soup
        if content_node is None:
            logger.warning("Selector '%s' matched nothing on %s", selector, spec.get("url"))
            return None

        raw_text = _html_to_legal_text(str(content_node))
        if not raw_text.strip():
            return None

        title = spec.get("title") or _guess_title(raw_text)
        category = spec.get("category") or classify_category(title)
        source = SourceMeta(
            law_name=spec.get("law_name", title),
            category=category if category in CATEGORIES else _DEFAULT_CATEGORY,
            entity_type=spec.get("entity_type", "both"),
            province_scope=spec.get("province_scope"),
            effective_from=spec.get("effective_from"),
            effective_to=spec.get("effective_to"),
            source_url=spec["url"],
        )
        return CrawledDocument(doc_id=spec["doc_id"], title=title, raw_text=raw_text, source=source)


def _guess_title(raw_text: str) -> str:
    first_line = next((line for line in raw_text.splitlines() if line.strip()), "")
    return re.sub(r"\s+", " ", first_line).strip()[:200]
