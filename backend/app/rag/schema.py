"""Shared data shapes for the retrieval pipeline.

These mirror the `tmquan/vbpl-vn` dataset schema directly (see
`app/ingestion/hf_dataset_loader.py`'s module docstring for the
confirmed field-by-field notes) rather than inventing a parallel
vocabulary — `scope`, `issuing_authority`, `doc_type`, `legal_area`,
`extracted_json`, `structure_json`, `markdown`, `confidence` all keep
the dataset's own names so a document read straight from the parquet
(or from whatever table `app/ingestion/pipeline.py` loads it into)
can be constructed into a `LegalDocument` with no relabeling step to
get wrong.

Two fields are worth flagging on first read:

  - `confidence` is carried because the schema has the column, but per
    the dataset build it is null on every row (see the caller's brief).
    Nothing in this package reads it as a trust signal — coverage.py
    builds its own score instead. It's kept on the dataclass purely so
    a future dataset build that *does* populate it doesn't require a
    schema change to use it.
  - `markdown` is `None` for ~7.2% of rows (metadata-only, no body).
    Every retrieval stage in this package treats `None` as "cannot be
    reached by lexical or embedding search" rather than filtering it
    out silently — see `coverage.py`'s `NULL_MARKDOWN` gap kind.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Scope = Literal["trung_uong", "dia_phuong"]


@dataclass(frozen=True)
class LegalDocument:
    """One row of the corpus, in the shape retrieval needs."""

    doc_name: str
    title: str
    doc_type: str  # canonical snake_case slug, e.g. "nghi_dinh"
    legal_type: str  # dataset's free-text instrument label, e.g. "Nghị định"
    scope: Scope
    issuing_authority: str
    doc_number: tuple[str, ...] = ()
    legal_area: Optional[str] = None  # "Chưa phân loại" for 71% — see jurisdiction.py note
    issue_date: Optional[str] = None
    markdown: Optional[str] = None  # None for the 7.2% metadata-only rows
    extracted_json: dict[str, Any] = field(default_factory=dict)  # carries entities/relations/statute_refs
    structure_json: dict[str, Any] = field(default_factory=dict)  # {"meta": {...}, "sections": [{"kind", "label", ...}], ...} — flat top-level "sections" list, confirmed via the dataset card's own quick-load example; see app/rag/obligation_retrieval.py's article-location note for the full detail
    confidence: Optional[float] = None  # NOT a trust signal — see module docstring
    source_url: Optional[str] = None
    van_ban_hop_nhat: Optional[str] = None  # doc_name of the consolidated text, when one exists

    @property
    def has_body(self) -> bool:
        return bool(self.markdown and self.markdown.strip())


@dataclass(frozen=True)
class JurisdictionFacets:
    """Jurisdiction is two facets, not one — see jurisdiction.py."""

    include_national: bool = True
    province: Optional[str] = None  # the user's own province, matched via issuing_authority


@dataclass(frozen=True)
class ArticleCitation:
    """A single Điều-level citation — the unit this pipeline cites at,
    per `structure_json`'s actual granularity (section/paragraph/
    sentence char-spans), never "see this whole document"."""

    doc_name: str
    title: str
    dieu_number: Optional[int]  # None only if structure_json had no article-level span for this hit
    excerpt: str
    char_span: tuple[int, int]
    doc_type: str
    issue_date: Optional[str]
    is_consolidated_text: bool  # True if this citation resolved to a văn bản hợp nhất


@dataclass(frozen=True)
class ConsolidationResult:
    """Output of consolidation.py's citation-graph walk (step 4)."""

    effective_doc_name: str  # the doc to actually cite — may differ from the doc we started at
    chain: tuple[str, ...]  # doc_names walked to reach it, starting doc included
    used_hop_nhat: bool  # True if a văn bản hợp nhất pointer was used instead of a graph walk
    confidence: float  # 0..1 — see consolidation.py; feeds coverage.py, not trusted alone
    unresolved: bool  # True if statute_refs pointed somewhere we couldn't follow (dead link, cycle, depth limit)


@dataclass(frozen=True)
class CoverageBreakdown:
    """Self-built coverage score (step 6) and the facets behind it —
    kept structured, not collapsed to a single float, so a caller can
    see *why* something scored low rather than just that it did."""

    jurisdiction_match: bool
    lexical_score: float  # 0..1
    embedding_score: Optional[float]  # 0..1, None if no embedding index available for this doc
    legal_area_boost: float  # 0..1 — a boost, see jurisdiction.py; never a filter
    hierarchy_confidence: float  # 0..1 — 1.0 for a ranked instrument type, lower for an unranked one
    consolidation_confidence: float  # 0..1, from ConsolidationResult
    markdown_available: bool
    total: float  # weighted combination — see coverage.py WEIGHTS


@dataclass(frozen=True)
class ObligationHit:
    """One retrieved obligation, ready to show the user."""

    citation: ArticleCitation
    hierarchy_rank: int  # lower = more authoritative, see hierarchy.py
    consolidation: ConsolidationResult
    coverage: CoverageBreakdown


GapKind = Literal["null_markdown_match", "low_coverage", "unresolved_consolidation"]


@dataclass(frozen=True)
class GapFlag:
    """A place the pipeline is explicitly telling the caller "don't
    trust that this obligation was captured" — the whole point of step
    6 is that these get surfaced, not silently dropped."""

    kind: GapKind
    doc_name: str
    title: str
    reason: str


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    facets: JurisdictionFacets
    obligations: list[ObligationHit]
    gaps: list[GapFlag]

    @property
    def has_unreviewed_gaps(self) -> bool:
        return len(self.gaps) > 0
