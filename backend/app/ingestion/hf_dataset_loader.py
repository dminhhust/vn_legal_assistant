"""Third `LegalSourceLoader`: streams real, already-crawled VBPL
documents from the `tmquan/vbpl-vn` dataset on Hugging Face, instead of
hitting the live VBPL gateway ourselves.

WHY THIS EXISTS (see app/ingestion/crawler.py's module docstring for
the other two loaders): `VbplGatewayCrawler` was confirmed, by
actually running it, to fail against the real gateway with
`400 Bad Request` on the very first list-page call. Digging into why
(via a real, independently-published dataset that crawled the same
source) turned up the actual root cause: `vbpl-bientap-gateway.moj.gov.vn`
is the backing API for vbpl.vn's single-page app, and it's gated
behind a Bearer token that the SPA obtains by solving Google's
invisible reCAPTCHA v2 in a real browser session. A plain
`requests.Session()` — which is all `VbplGatewayCrawler` uses — has no
way to obtain that token, so every request is rejected outright. This
is a harder blocker than the "field names might drift" caveat
`VbplGatewayCrawler`'s docstring originally flagged, and solving it
properly needs headless-browser + captcha-token automation, which is
out of scope here (see the honesty note in crawler.py, updated
alongside this file, for the full explanation).

`tmquan/vbpl-vn` (https://huggingface.co/datasets/tmquan/vbpl-vn,
CC-BY-4.0) already did that crawl properly — 158,822 real Vietnamese
legal documents from the same official vbpl.vn source, Ministry of
Justice, with clean parsed body text and a documented schema (doc
type, issue date, issuing authority, legal area, and a structured
entity/citation layer). Streaming from it gets a real ingested corpus
into this app today without needing to solve the auth problem above.
It is NOT a substitute for a live crawler long-term (it's a
point-in-time mirror, captured 2026-05-23T14:29:39Z per its dataset
card — see "Loading legal data" in the README for how to keep this
current), but it's real government-published legal text, not
synthetic fixture data, and its license permits this redistribution.

SCHEMA NOTES (confirmed against the dataset card, not guessed):
  - `markdown` is null for 11,505/158,822 rows (7.2%) — legacy
    documents vbpl.vn itself no longer carries a body for
    (`body_source == "shell_html"`). Skipped; nothing to ingest.
  - `doc_number` is a LIST, not a single string — a minority of rows
    pack multiple identifiers (e.g. an amendment citing two prior
    decisions). All elements are kept, not just the first.
  - `doc_type` is one of 25 canonical snake_case slugs (in this build)
    from a larger fixed enumeration — confirmed against
    `CANONICAL_CODE_TO_SLUG` in the dataset's own build pipeline
    (https://github.com/tmquan/ViLA/blob/main/packages/datasites/vbpl/codes.py),
    not guessed from the dataset card's top-12 counts table alone.
    Of that enumeration, the ones NOT "văn bản quy phạm pháp luật"
    (binding normative legal instruments) under *Luật Ban hành Văn bản
    Quy phạm Pháp luật 2015* — translations, correspondence, notices,
    international agreements/protocols/MOUs, and undifferentiated
    "other/related" catch-alls — are excluded by default (see
    `excluded_doc_types`) rather than ingested as if they were primary
    legal text. `sac_lenh`/`sac_luat` (historical decrees) and
    `nghi_quyet_lien_tich`/`thong_tu_lien_bo` (joint/inter-ministerial
    instruments outside the dataset card's top-12 table but present in
    the full slug enumeration) ARE normative and are deliberately kept.
  - `legal_area` is the source portal's own subject-area tag (~250
    distinct values) but is `"Chưa phân loại"` (uncategorised) for
    71.0% of rows. Where it IS populated, it's a stronger category
    signal than the title alone — folded into this app's title-based
    `classify_category()` (see `_classify()` below) rather than
    building a separate ~250-entry mapping table, since
    `classify_category`'s existing keyword list ("thuế", "lao động",
    "đất đai", ...) already substring-matches most `legal_area` values
    directly (e.g. `legal_area="Quản lý thuế, phí và lệ phí"` contains
    "thuế" and "phí").
  - `scope` is `dia_phuong` (provincial, 65.7%) or `trung_uong`
    (central, 34.3%) but does not itself name the province.
    `issuing_authority` usually does (e.g. `"UBND tỉnh Bà Rịa - Vũng
    Tàu"`, `"HĐND Tỉnh Phú Thọ"`) — extracted into
    `SourceMeta.province_scope` on a best-effort basis (see
    `_extract_province()`); left `None` (→ "national") when no
    "tỉnh"/"thành phố" pattern is found, rather than guessed.

Requires the `datasets` package (see requirements.txt) and network
access to huggingface.co, which — like the live gateway — this
sandbox's own egress allowlist does not include; this has not been
exercised from here for the same reason `VbplGatewayCrawler` couldn't
be. Run it from an environment with real network access (the same one
where you confirmed the gateway crawl fails) and spot-check the first
few ingested documents by hand, exactly as recommended for the other
two loaders.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

from app.ingestion.crawler import CrawledDocument, classify_category
from app.ingestion.metadata import CATEGORIES, SourceMeta

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "tmquan/vbpl-vn"
_DEFAULT_CATEGORY = "residence_civil"  # matches crawler.py's fallback bucket

# The full doc_type slug enumeration (confirmed via
# `CANONICAL_CODE_TO_SLUG` in codes.py — see module docstring) is
# wider than the dataset card's top-12 table shows. Beyond the two
# originally identified here (`ban_dich_van_ban`, `cong_van`), the
# remaining non-normative slugs in that enumeration are administrative
# or diplomatic instruments never on Luật Ban hành Văn bản Quy phạm
# Pháp luật 2015's Điều 4 list either — they aren't binding legal text
# themselves, just auxiliary/related material, so all of them are
# excluded from ingestion by default:
#   - ban_dich_van_ban  — a TRANSLATION of another document (6.7% of rows)
#   - cong_van          — an official dispatch/correspondence (0.4%)
#   - thong_bao         — a notice/announcement
#   - hiep_dinh         — an international agreement/treaty (governed
#                          by treaty law, not this domestic VBQPPL list)
#   - nghi_dinh_thu     — a protocol (usually to an international agreement)
#   - ban_ghi_nho       — a memorandum of understanding (non-binding)
#   - thoa_thuan        — an agreement/arrangement (non-binding)
#   - van_ban_hanh_chinh_lien_quan — literally "related administrative document"
#   - van_ban_khac      — literally "other document" (undifferentiated catch-all)
#   - van_ban_lien_quan — literally "related document" (undifferentiated catch-all)
#   - chuong_trinh      — a programme/scheme document, not a legal instrument
#   - chua_xac_dinh     — literally "undetermined" (unclassified on the source portal)
# `sac_lenh`, `sac_luat`, `nghi_quyet_lien_tich`, and `thong_tu_lien_bo`
# are deliberately NOT here — all four are real (if some historical)
# normative instruments; see app/rag/hierarchy.py for how they're
# ranked once ingested. Overridable via the constructor for callers
# who want any of these anyway (e.g. `ban_dich_van_ban` may be useful
# for an English-language variant of this app).
DEFAULT_EXCLUDED_DOC_TYPES: frozenset[str] = frozenset(
    {
        "ban_dich_van_ban",
        "cong_van",
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
    }
)

# `issuing_authority` for scope="dia_phuong" rows is almost always
# "<UBND|HĐND> [tỉnh|Tỉnh|thành phố|Thành phố] <province/city name>"
# (confirmed against the dataset card's top-15 issuing-agency table:
# "UBND tỉnh Bà Rịa - Vũng Tàu", "HĐND Tỉnh Phú Thọ", "UBND Thành phố
# Hồ Chí Minh", ...). Extracts just the province/city name. Best-
# effort: an issuing_authority that doesn't follow this pattern (rare
# — central agencies like "Bộ Tài chính" naturally never match, which
# is correct; but an unusual provincial body naming style could also
# fail to match) simply leaves province_scope unset rather than
# guessing wrong.
_PROVINCE_RE = re.compile(r"(?:tỉnh|thành\s*phố)\s+(.+)$", re.IGNORECASE)


def _extract_province(issuing_authority: Optional[str]) -> Optional[str]:
    if not issuing_authority:
        return None
    m = _PROVINCE_RE.search(issuing_authority)
    return m.group(1).strip() if m else None


def _classify(title: str, legal_area: Optional[str]) -> str:
    """Category classification informed by BOTH signals the dataset
    actually provides. `legal_area` is the source portal's own subject
    tag and, where populated, is more reliable than inferring purely
    from `title` — but it's `"Chưa phân loại"` (uncategorised) for
    71.0% of rows, so title alone is the only signal for most of the
    corpus anyway. Deliberately reuses crawler.py's existing
    substring-keyword `classify_category()` against the combined text
    rather than hand-building a second ~250-entry legal_area->category
    table: `legal_area` values are themselves substring-matched by the
    same keywords ("thuế", "lao động", "đất đai", ...) that already
    classify titles, e.g. `legal_area="Quản lý thuế, phí và lệ phí"`
    contains "thuế" and "phí" and gets picked up for free."""
    is_tagged = bool(legal_area) and legal_area != "Chưa phân loại"
    text = f"{legal_area} {title}" if is_tagged else title
    return classify_category(text)


# app.ingestion.parser's _CHAPTER_RE/_ARTICLE_RE are line-anchored (`^...`,
# matched per `raw_text.splitlines()` line) — they require "Chương ..."/
# "Điều N" to be the first thing on its own line. tmquan/vbpl-vn's
# `markdown` field does NOT guarantee that: it was produced by generic
# `markdownify` HTML->text conversion (not this app's own
# app.ingestion.crawler._html_to_legal_text, which deliberately turns
# block elements into newlines), and for many rows collapses an entire
# document into one flowing paragraph with "Điều 1. ... Điều 2. ..."
# appearing inline, never at a line start. Confirmed empirically: rows
# whose own `extracted_json` shows multiple "Điều N" entities
# nonetheless report `num_paragraphs: 1` in the same row's stats — i.e.
# the dataset's own structure extractor saw the same run-on text we do,
# it just uses offset-based (not line-based) matching so it wasn't
# broken by it the way our line-anchored parser is.
#
# Fix, part 1: force a line break immediately before every "Chương ..."
# occurrence, wherever it lands, so parser.py's line-anchored regex can
# find it. `(?<!\n)` avoids doubling up a newline that's already there.
_CHAPTER_TOKEN_RE = re.compile(r"(?<!\n)(Ch(?:ươ|uơ)ng\s+(?:[IVXLCDM]+|\d+)\.?)", re.IGNORECASE)

# Fix, part 2: "Điều N" needs the same treatment, but naively line-
# breaking on EVERY occurrence is wrong and was confirmed wrong in
# production: Vietnamese legal text very commonly CITES another
# document's article inline — "... quy định tại Điều 8 của Luật Ngân
# sách nhà nước ...", "sửa đổi Điều 3 Nghị định số ..." — which is not
# a heading of THIS document at all. Treating every citation as a new
# heading produced a real DuplicateIDError: two "Điều 8" occurrences in
# one document (a real Điều 8 heading, plus a citation to a different
# law's Điều 8), both hashed to the same chunk id.
#
# Two independent signals gate whether a match is accepted as a
# genuine heading, since neither alone is reliable:
#
#  1. STRICT ASCENDING NUMBER — real Vietnamese legal drafting always
#     numbers its own articles in strict ascending order; a citation
#     can reference any number, so anything that doesn't advance the
#     sequence is rejected. Not sufficient alone: if a citation's
#     number happens to equal (or exceed) the real next article's
#     number and appears just before it in text — a realistic pattern,
#     since documents often cite a related provision right where
#     they're about to state their own parallel one — this check alone
#     would accept the CITATION as the heading and then reject the
#     real heading that follows for failing to advance further.
#  2. NOT IMMEDIATELY PRECEDED BY A CITATION PREPOSITION — "tại Điều",
#     "theo Điều", "quy định tại Điều", "của Luật/Nghị định ... Điều"
#     are the standard Vietnamese citation phrasings, and a genuine
#     heading is never preceded by one (it starts its own sentence).
#     Checked via a short lookback window immediately before the match.
#
# A match must pass BOTH to be treated as a heading; anything else is
# left inline as ordinary body text (folded into whichever article's
# body it appears in), never treated as a boundary.
#
# Known residual limitation: this is a heuristic, not a citation
# parser — an unusual citation phrasing this list doesn't cover, whose
# number also happens to advance the sequence, could still be
# misread as a heading. parser.py already drops undetected preamble
# content by design, so the failure mode stays "loses a document's
# real structure in an unusual phrasing" rather than "crashes or
# corrupts other documents" — not attempting to solve this with a full
# citation parser for this MVP.
_ARTICLE_TOKEN_RE = re.compile(r"Điều\s+(\d+)\.?")
_CITATION_PRECEDED_BY_RE = re.compile(
    r"(?:tại|theo|của|sửa\s*đổi|bổ\s*sung|bãi\s*bỏ|hủy\s*bỏ|thay\s*thế|nêu\s*ở|quy\s*định\s*ở)\s*$",
    re.IGNORECASE,
)
_CITATION_LOOKBACK_CHARS = 40


def _normalize_markdown_for_parser(markdown: str) -> str:
    text = _CHAPTER_TOKEN_RE.sub(r"\n\1", markdown)
    return _insert_article_linebreaks(text)


def _insert_article_linebreaks(text: str) -> str:
    last_number = 0

    def repl(m: re.Match) -> str:
        nonlocal last_number
        num = int(m.group(1))
        start = m.start()
        preceding = text[max(0, start - _CITATION_LOOKBACK_CHARS):start]
        is_citation = _CITATION_PRECEDED_BY_RE.search(preceding) is not None
        if num <= last_number or is_citation:
            return m.group(0)  # citation/reference — leave inline, not a boundary
        last_number = num
        needs_newline = start > 0 and text[start - 1] != "\n"
        return ("\n" if needs_newline else "") + m.group(0)

    return _ARTICLE_TOKEN_RE.sub(repl, text)


@dataclass
class HfVbplDatasetLoader:
    """Streams rows from `tmquan/vbpl-vn` and yields them in the same
    `CrawledDocument` shape the other two loaders produce, so
    run_ingestion.py can drive any of the three interchangeably.

    Uses `streaming=True` deliberately — this is a 3.86GB / 158K-row
    parquet dataset, and a full local download is never necessary for
    an MVP's `--max-documents` smoke run.
    """

    dataset_name: str = DEFAULT_DATASET_NAME
    split: str = "train"
    category_overrides: dict[str, str] = field(default_factory=dict)  # doc_name -> category
    # See DEFAULT_EXCLUDED_DOC_TYPES above for why these two are
    # excluded by default — not "văn bản quy phạm pháp luật" under the
    # law this dataset's own doc_type taxonomy follows.
    excluded_doc_types: frozenset[str] = field(default_factory=lambda: DEFAULT_EXCLUDED_DOC_TYPES)
    # Injectable row source — same "provider abstraction" pattern as
    # crawler.py's HttpClient injection, so this loader is fully
    # testable offline against fixture rows (see
    # tests/test_ingestion_hf_dataset_loader.py) without needing live
    # network access to huggingface.co. Defaults to the real streaming
    # HF dataset load; tests override it with a fake iterable.
    rows_provider: Optional[Callable[[], Iterable[dict]]] = None

    def _default_rows_provider(self) -> Iterable[dict]:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - dependency documented in requirements.txt
            raise RuntimeError(
                "the 'datasets' package is required for HfVbplDatasetLoader "
                "(pip install datasets)"
            ) from exc

        return load_dataset(self.dataset_name, split=self.split, streaming=True)

    def crawl(self, *, max_documents: Optional[int] = None) -> Iterator[CrawledDocument]:
        rows = (self.rows_provider or self._default_rows_provider)()

        yielded = 0
        for row in rows:
            if max_documents is not None and yielded >= max_documents:
                return

            doc = self._row_to_document(row)
            if doc is None:
                continue

            yield doc
            yielded += 1

    def _row_to_document(self, row: dict) -> Optional[CrawledDocument]:
        markdown = row.get("markdown")
        title = row.get("title")
        doc_name = row.get("doc_name")

        # 7.2% of the dataset has markdown=null (confirmed exact figure
        # from the dataset card, body_source=="shell_html") — legacy
        # documents the official source itself no longer carries a body
        # for. Nothing to ingest.
        if not markdown or not markdown.strip():
            return None
        if not title or not doc_name:
            return None

        doc_type = row.get("doc_type")
        if doc_type in self.excluded_doc_types:
            return None

        raw_text = _normalize_markdown_for_parser(markdown)

        # doc_number is a LIST — a minority of rows carry more than one
        # identifier (e.g. an amendment citing two prior decisions by
        # number). Keep all of them, not just the first.
        doc_numbers = row.get("doc_number") or []
        doc_number = "; ".join(doc_numbers) if doc_numbers else None
        legal_type = row.get("legal_type") or ""
        law_name = f"{legal_type} {doc_number} - {title}".strip(" -") if doc_number else f"{legal_type} {title}".strip()

        legal_area = row.get("legal_area")
        category = self.category_overrides.get(doc_name) or _classify(title, legal_area)
        if category not in CATEGORIES:
            category = _DEFAULT_CATEGORY

        province_scope = None
        if row.get("scope") == "dia_phuong":
            province_scope = _extract_province(row.get("issuing_authority"))

        source = SourceMeta(
            law_name=law_name,
            category=category,
            entity_type="both",
            province_scope=province_scope,
            effective_from=row.get("issue_date"),
            source_url=row.get("source_url"),
        )
        return CrawledDocument(
            doc_id=f"vbpl-hf-{doc_name}",
            title=title,
            raw_text=raw_text,
            source=source,
        )
