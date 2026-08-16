"""Step 4 — the consolidation walk.

There's no "current version" field in the dataset, so recency isn't a
date filter here — it's a graph walk:

  1. If the document already has a văn bản hợp nhất (consolidated
     text) — `van_ban_hop_nhat` — prefer it outright. This exists for
     only ~1.1% of docs, but where it exists it's vbpl.vn's OWN
     curated consolidation, not an inferred one, so it's trusted at
     high confidence.
  2. Otherwise, walk `extracted_json["statute_refs"]` for amend/
     replace/repeal relationships, following the most recent
     supersession at each hop, up to `max_hops`.

CRITICAL CAVEAT, per the caller's brief: statute_refs is NOT an
official cross-reference table — it comes from a regex/dictionary
extractor over the document text, so it can miss a relationship or
misattribute one. This module does NOT treat a walk result as
ground truth. Every walk returns a `confidence` in [0, 1] (1.0 only
for the no-walk-needed and văn-bản-hợp-nhất cases) that DECAYS with
each inferred hop, and that confidence is a direct input to
coverage.py's step-6 scoring — a shaky consolidation walk should pull
a result's coverage score down, not silently present a wrong "current"
version with the same confidence as a directly-published one.

A repeal with no replacement, a cycle, or a statute_ref pointing to a
doc_name this corpus doesn't have are all treated as `unresolved=True`
rather than swallowed — obligation_retrieval.py surfaces these as
`unresolved_consolidation` gaps (step 6) instead of just returning
whatever the walk got to.
"""
from __future__ import annotations

from typing import Callable, Optional

from app.rag.schema import ConsolidationResult, LegalDocument

# Relationship kinds statute_refs entries may carry, and how each
# affects the walk. "repeal" with no follow-up target means the
# obligation may no longer be current at all — that's exactly the
# kind of thing that must not be silently dropped, so a bare repeal
# with nothing replacing it is treated as unresolved rather than as
# "stop here, this is fine."
_SUPERSEDING_RELATIONS = {"amend", "replace", "sua_doi", "thay_the"}
_REPEAL_RELATIONS = {"repeal", "bai_bo", "huy_bo"}

# Regex/dictionary extraction is inherently noisier per hop; each
# inferred hop discounts confidence multiplicatively. Tuned to be
# conservative (a 3-hop inferred chain lands around 0.5, well below
# the coverage-score threshold that would let it pass unflagged) since
# the brief explicitly asks for missed/wrong links to be caught, not
# smoothed over.
_PER_HOP_CONFIDENCE_DECAY = 0.75
_HOP_NHAT_CONFIDENCE = 0.95  # vbpl.vn-curated, not extractor-inferred — still short of 1.0: still a pointer, not a guarantee it's this exact query's applicable text
_NO_WALK_CONFIDENCE = 1.0  # the document is its own current text; nothing was inferred


def _statute_refs(doc: LegalDocument) -> list[dict]:
    refs = doc.extracted_json.get("statute_refs") or []
    return [r for r in refs if isinstance(r, dict)]


def consolidate(
    doc: LegalDocument,
    resolve: Callable[[str], Optional[LegalDocument]],
    *,
    max_hops: int = 5,
) -> ConsolidationResult:
    """`resolve(doc_name) -> LegalDocument | None` is injected (same
    provider-abstraction pattern the rest of this codebase uses for
    HTTP clients / embedders) so this is testable against a fixed
    fixture graph without a real document store."""

    if doc.van_ban_hop_nhat:
        consolidated = resolve(doc.van_ban_hop_nhat)
        if consolidated is not None:
            return ConsolidationResult(
                effective_doc_name=consolidated.doc_name,
                chain=(doc.doc_name, consolidated.doc_name),
                used_hop_nhat=True,
                confidence=_HOP_NHAT_CONFIDENCE,
                unresolved=False,
            )
        # van_ban_hop_nhat pointer exists but target isn't in the
        # corpus we can reach — don't silently fall through to the
        # (possibly superseded) original as if nothing were wrong.
        return ConsolidationResult(
            effective_doc_name=doc.doc_name,
            chain=(doc.doc_name,),
            used_hop_nhat=False,
            confidence=0.0,
            unresolved=True,
        )

    chain = [doc.doc_name]
    current = doc
    confidence = _NO_WALK_CONFIDENCE
    seen = {doc.doc_name}

    for _ in range(max_hops):
        refs = _statute_refs(current)
        superseding = [r for r in refs if r.get("relation") in _SUPERSEDING_RELATIONS]
        repealing = [r for r in refs if r.get("relation") in _REPEAL_RELATIONS]

        if not superseding and not repealing:
            break  # nothing further found — current is the walk's end, at whatever confidence we've accumulated

        if repealing and not superseding:
            # Repealed with nothing shown replacing it — this is
            # exactly the "obligation might have vanished" case the
            # brief warns about. Don't guess; flag it.
            return ConsolidationResult(
                effective_doc_name=current.doc_name,
                chain=tuple(chain),
                used_hop_nhat=False,
                confidence=confidence,
                unresolved=True,
            )

        target_name = superseding[0].get("target_doc_name")
        ref_confidence = superseding[0].get("confidence")
        hop_confidence = ref_confidence if isinstance(ref_confidence, (int, float)) else _PER_HOP_CONFIDENCE_DECAY

        if not target_name or target_name in seen:
            # Missing target or a cycle back to something already
            # walked — stop rather than loop or dereference nothing.
            return ConsolidationResult(
                effective_doc_name=current.doc_name,
                chain=tuple(chain),
                used_hop_nhat=False,
                confidence=confidence,
                unresolved=not target_name or target_name in seen,
            )

        target_doc = resolve(target_name)
        if target_doc is None:
            # statute_refs points somewhere this corpus doesn't have —
            # a dead link. Surface it, don't pretend `current` is final.
            return ConsolidationResult(
                effective_doc_name=current.doc_name,
                chain=tuple(chain),
                used_hop_nhat=False,
                confidence=confidence,
                unresolved=True,
            )

        confidence *= hop_confidence
        chain.append(target_name)
        seen.add(target_name)
        current = target_doc

    else:
        # Exhausted max_hops without settling — long inferred chains
        # are exactly where a regex extractor is least trustworthy.
        return ConsolidationResult(
            effective_doc_name=current.doc_name,
            chain=tuple(chain),
            used_hop_nhat=False,
            confidence=confidence,
            unresolved=True,
        )

    return ConsolidationResult(
        effective_doc_name=current.doc_name,
        chain=tuple(chain),
        used_hop_nhat=False,
        confidence=confidence,
        unresolved=False,
    )
