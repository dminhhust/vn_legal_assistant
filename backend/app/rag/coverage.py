"""Step 6 — coverage scoring and gap detection.

The dataset's own `confidence` column is populated in the schema but
null on every row in this build, so it cannot be reused as a trust
signal — this module builds one from scratch, out of match strength
across the facets the rest of the pipeline already computed:
jurisdiction match, lexical score, embedding similarity, the
legal_area boost, hierarchy ranking confidence, and consolidation-walk
confidence. None of these alone is "confidence"; the combination is a
substitute for a signal the data doesn't give us, so it should not be
oversold as more precise than that. It's used for two purposes only:
ranking (higher total = show first) and gap-flagging (below
threshold = flag for manual review, never silently drop).

WHY GAPS ARE A FIRST-CLASS OUTPUT, NOT A LOGGING DETAIL: the caller's
brief is explicit that the point of this redesign is "no obligation
missed." Three situations can make a real, applicable obligation
absent from `obligations` even when the pipeline is working correctly,
and all three need to surface as `GapFlag`s rather than disappear:

  1. `null_markdown_match` — a document that matches the user's
     jurisdiction (and, if legal_area happens to be populated, the
     query's topic) but has `markdown is None`. It CANNOT be found by
     lexical or embedding search (there's no text to search), so it
     will never appear as a search hit no matter how good the ranking
     is. obligation_retrieval.py has to query for these separately by facet match
     alone and hand them to this module to flag — see
     `flag_null_markdown_matches` below.
  2. `low_coverage` — a hit came back from search but scored below
     `MIN_COVERAGE_TO_TRUST`. Still shown as a gap (not dropped
     outright) so a human reviewing results sees "something matched,
     weakly" rather than nothing.
  3. `unresolved_consolidation` — the consolidation walk (step 4)
     couldn't establish a confident current version (dead statute_ref,
     cycle, unresolved repeal, hop-limit exhaustion).
"""
from __future__ import annotations

from typing import Iterable

from app.rag.hierarchy import is_ranked
from app.rag.schema import (
    ConsolidationResult,
    CoverageBreakdown,
    GapFlag,
    JurisdictionFacets,
    LegalDocument,
)
from app.rag.jurisdiction import legal_area_boost, matches_jurisdiction

# Weights sum to 1.0. jurisdiction_match is deliberately the largest
# single weight and is ALSO enforced as a hard filter upstream
# (jurisdiction.matches_jurisdiction) — belt and suspenders, since a
# jurisdiction mistake is a wrong-obligation mistake, not just a
# ranking wobble. legal_area_boost gets the smallest weight precisely
# because it's a boost, not primary recall — see jurisdiction.py.
WEIGHTS = {
    "jurisdiction_match": 0.30,
    "lexical_score": 0.20,
    "embedding_score": 0.20,
    "legal_area_boost": 0.05,
    "hierarchy_confidence": 0.10,
    "consolidation_confidence": 0.15,
}

# Below this total, a hit is still returned but wrapped in a
# `low_coverage` GapFlag rather than presented as a trusted result.
# Deliberately conservative — false "flag it" costs a human a second
# look; false "don't flag it" costs a missed obligation, which is the
# one thing this redesign is explicitly trying not to do.
MIN_COVERAGE_TO_TRUST = 0.6


def score(
    doc: LegalDocument,
    facets: JurisdictionFacets,
    *,
    lexical_score: float,
    embedding_score: float | None,
    query_keywords: list[str],
    consolidation: ConsolidationResult,
) -> CoverageBreakdown:
    jurisdiction_ok = matches_jurisdiction(doc, facets)
    area_boost = legal_area_boost(doc, query_keywords)
    hierarchy_conf = 1.0 if is_ranked(doc.doc_type, doc.scope) else 0.5

    total = (
        WEIGHTS["jurisdiction_match"] * (1.0 if jurisdiction_ok else 0.0)
        + WEIGHTS["lexical_score"] * lexical_score
        + WEIGHTS["embedding_score"] * (embedding_score if embedding_score is not None else 0.0)
        + WEIGHTS["legal_area_boost"] * area_boost
        + WEIGHTS["hierarchy_confidence"] * hierarchy_conf
        + WEIGHTS["consolidation_confidence"] * consolidation.confidence
    )

    return CoverageBreakdown(
        jurisdiction_match=jurisdiction_ok,
        lexical_score=lexical_score,
        embedding_score=embedding_score,
        legal_area_boost=area_boost,
        hierarchy_confidence=hierarchy_conf,
        consolidation_confidence=consolidation.confidence,
        markdown_available=doc.has_body,
        total=total,
    )


def flag_low_coverage(doc: LegalDocument, breakdown: CoverageBreakdown) -> GapFlag | None:
    if breakdown.total >= MIN_COVERAGE_TO_TRUST:
        return None
    return GapFlag(
        kind="low_coverage",
        doc_name=doc.doc_name,
        title=doc.title,
        reason=(
            f"Combined coverage score {breakdown.total:.2f} is below the "
            f"{MIN_COVERAGE_TO_TRUST:.2f} trust threshold "
            f"(jurisdiction_match={breakdown.jurisdiction_match}, "
            f"lexical={breakdown.lexical_score:.2f}, "
            f"embedding={breakdown.embedding_score}, "
            f"legal_area_boost={breakdown.legal_area_boost:.2f}, "
            f"hierarchy_confidence={breakdown.hierarchy_confidence:.2f}, "
            f"consolidation_confidence={breakdown.consolidation_confidence:.2f})."
        ),
    )


def flag_unresolved_consolidation(doc: LegalDocument, consolidation: ConsolidationResult) -> GapFlag | None:
    if not consolidation.unresolved:
        return None
    return GapFlag(
        kind="unresolved_consolidation",
        doc_name=doc.doc_name,
        title=doc.title,
        reason=(
            "Citation-graph walk could not confidently establish the current "
            f"version of this document (chain so far: {' -> '.join(consolidation.chain)}). "
            "The document as retrieved may have been amended, replaced, or repealed "
            "by something the statute_refs extractor missed or mis-linked."
        ),
    )


def flag_null_markdown_matches(
    candidates: Iterable[LegalDocument],
    facets: JurisdictionFacets,
    query_keywords: list[str],
) -> list[GapFlag]:
    """The dedicated pass for the class of gap that NOTHING upstream
    can catch on its own: a document with `markdown is None` never
    produces a lexical or embedding hit, so it never enters the normal
    ranking path at all. `candidates` here must come from a facet-only
    query against the corpus (jurisdiction + optionally legal_area),
    not from search results — see obligation_retrieval.py's `_null_markdown_gaps`
    for how the doc store is expected to expose that query."""
    flags: list[GapFlag] = []
    for doc in candidates:
        if doc.has_body:
            continue
        if not matches_jurisdiction(doc, facets):
            continue
        area_boost = legal_area_boost(doc, query_keywords)
        area_untagged = not doc.legal_area or doc.legal_area == "Chưa phân loại"
        # Include it if the topic boost hit, OR legal_area is simply
        # untagged (71% of the corpus) — an untagged doc is not
        # evidence of irrelevance, so it can't be excluded on that
        # basis alone; it's exactly the kind of gap this pass exists
        # to surface rather than silently reason away.
        if area_boost > 0 or area_untagged:
            flags.append(
                GapFlag(
                    kind="null_markdown_match",
                    doc_name=doc.doc_name,
                    title=doc.title,
                    reason=(
                        "This document matches the user's jurisdiction but has no body "
                        "text (markdown is null — metadata-only row) and therefore cannot "
                        "be reached by lexical or embedding search. Its title/metadata "
                        "suggest it may be relevant; it needs manual review to confirm."
                    ),
                )
            )
    return flags
