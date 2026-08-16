"""Jurisdiction facets and the legal_area boost.

Two things this module deliberately does NOT do, both because the
dataset shape rules them out:

1. Treat jurisdiction as a single facet. `scope` (trung_uong /
   dia_phuong) tells you central-vs-provincial, but a provincial
   document only binds ITS province — `scope="dia_phuong"` alone can't
   distinguish a Hanoi decision from a Cần Thơ one. `issuing_authority`
   is what actually names the province (e.g. "UBND tỉnh Bà Rịa - Vũng
   Tàu"). So jurisdiction here is genuinely two facets combined:
   national law (scope=="trung_uong", always in scope, no user input
   needed) plus provincial law that matches the user's own province
   (scope=="dia_phuong" AND issuing_authority names their province).
   `matches_jurisdiction()` is a hard filter on this — a document
   outside both facets legally does not apply to the user and should
   not be retrievable no matter how well it matches the query text.

2. Filter on `legal_area`. It's ~520 distinct values and "Chưa phân
   loại" (uncategorised) on 71% of rows — filtering on it as a facet
   would silently drop most of the corpus, including for users whose
   actual obligation sits in an uncategorised document. `legal_area`
   is used ONLY as a boost (`legal_area_boost()`) on top of whatever
   full-text/embedding search already found, per the caller's brief:
   "full-text/embedding search carrying the real recall."
"""
from __future__ import annotations

import re
import unicodedata

from app.rag.schema import JurisdictionFacets, LegalDocument

_PROVINCE_PREFIX_RE = re.compile(r"^(?:tỉnh|thành\s*phố)\s+", re.IGNORECASE)


def resolve_facets(user_province: str | None) -> JurisdictionFacets:
    """National law is always in scope; provincial law is in scope
    only for the user's own province, if they've given one. A user who
    hasn't told us their province still gets full national-law
    coverage — they just won't see provincial obligations, which is
    correct (we don't know which province's rules would apply to
    them) rather than a bug to work around by guessing."""
    return JurisdictionFacets(include_national=True, province=user_province)


def _normalize_province(name: str) -> str:
    """NFC-normalize, strip the tỉnh/thành phố prefix, casefold. Used
    on both sides of a province comparison so "tỉnh Phú Thọ" and "Phú
    Thọ" match, without needing a canonical 63-province lookup table
    (which the dataset doesn't ship either)."""
    normalized = unicodedata.normalize("NFC", name).strip()
    normalized = _PROVINCE_PREFIX_RE.sub("", normalized)
    return normalized.casefold()


def _issuing_authority_names_province(issuing_authority: str, province: str) -> bool:
    normalized_authority = unicodedata.normalize("NFC", issuing_authority).casefold()
    normalized_province = _normalize_province(province)
    return normalized_province in normalized_authority


def matches_jurisdiction(doc: LegalDocument, facets: JurisdictionFacets) -> bool:
    """Hard filter — see module docstring point 1. Not a boost:
    a document that fails this can never be part of an obligation
    result, regardless of match strength elsewhere in the pipeline."""
    if doc.scope == "trung_uong":
        return facets.include_national
    if doc.scope == "dia_phuong":
        if facets.province is None:
            return False
        return _issuing_authority_names_province(doc.issuing_authority, facets.province)
    # Unrecognised scope value — fail closed (not a match) rather than
    # silently including a document we can't place jurisdictionally.
    return False


def legal_area_boost(doc: LegalDocument, query_keywords: list[str]) -> float:
    """A soft boost in [0, 1.0], never a filter — see module
    docstring point 2. Returns 0.0 (no boost, not a penalty) whenever
    `legal_area` is missing/uncategorised, so an uncategorised
    document scores the same as one that simply has no legal_area
    signal at all; it's never treated as evidence AGAINST relevance.
    """
    if not doc.legal_area or doc.legal_area == "Chưa phân loại":
        return 0.0
    area_lower = doc.legal_area.casefold()
    if any(kw.casefold() in area_lower for kw in query_keywords if kw):
        return 1.0
    return 0.0
