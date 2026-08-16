"""Statutory instrument hierarchy, per Luật Ban hành Văn bản Quy phạm
Pháp luật 2015 (Điều 4 — the legal-effect-order article).

NOT IN THE DATASET. `tmquan/vbpl-vn` gives `doc_type` as a flat
snake_case slug set (25 distinct values in this build) with no
precedence field, and vbpl.vn itself doesn't expose one either — this
map is hand-built from the 2015 Law's own Điều 4 ordering, not derived
from the data. Treat it the way this codebase already treats
hand-built mappings elsewhere (e.g. hf_dataset_loader.py's province
regex): best-effort, spot-checkable, and explicit about what it
doesn't cover rather than silently guessing.

The slug set itself is confirmed against the dataset's own build
pipeline — `CANONICAL_CODE_TO_SLUG` in
https://github.com/tmquan/ViLA/blob/main/packages/datasites/vbpl/codes.py
— rather than guessed from the top-12 counts shown on the dataset
card. That source is what caught the two real bugs this version of
the map fixes (see the inline comments below):

  1. There is no "nghi_quyet_qh" slug anywhere in the dataset. An
     earlier version of this map invented one to represent National
     Assembly resolutions distinctly from provincial HĐND ones — it
     never matched a single real row (the real column value is always
     the single slug "nghi_quyet" for BOTH), so every nghị quyết in
     the corpus (16.4% of it — the second most common doc_type)
     silently fell through to `UNRANKED_FALLBACK` regardless of its
     actual authority. Fixed below via a `scope`-aware special case,
     since `scope` is the one other column that can help disambiguate
     the two.
  2. `sac_lenh` (Sắc lệnh — a real, if historical, normative
     instrument: 980 rows, 0.6% of the corpus) was entirely absent
     from the map, also falling to `UNRANKED_FALLBACK`. Added at the
     Lệnh/Pháp lệnh tier, its real-world equivalent. `sac_luat` (Sắc
     luật) and `nghi_quyet_lien_tich` (present in the full slug
     enumeration even though not in the dataset card's top-12 table)
     were missing for the same reason and are added too.

WHY THIS MATTERS FOR RETRIEVAL: without an explicit rank, two
documents that both look like a strong lexical/embedding match for the
same query can silently get treated as equally authoritative — e.g. a
quyết định (decision) issued by a provincial People's Committee
surfacing above the luật (law) it implements just because it happens
to use closer wording to the user's question. `rank_of()` exists so
obligation_retrieval.py can sort by legal authority first and match strength
second, never the other way around.
"""
from __future__ import annotations

# Lower number = higher legal authority (binds documents ranked below
# it; cannot be overridden by them). Ties (same rank) share authority
# and are ordered by match strength only within that tier.
#
# Grouped per Điều 4: Hiến pháp > Luật/Nghị quyết của QH > Pháp
# lệnh/Nghị quyết của UBTVQH > văn bản của Chủ tịch nước > Nghị định >
# Thông tư/Thông tư liên tịch > văn bản của HĐND/UBND cấp tỉnh, and so
# on down to commune level. `tmquan/vbpl-vn`'s doc_type slugs don't all
# map 1:1 onto Điều 4's own list (e.g. it has no separate "văn bản Chủ
# tịch nước" slug), so this collapses to the slugs actually present in
# the dataset (per `CANONICAL_CODE_TO_SLUG` in codes.py — see module
# docstring), ordered by where their real-world instrument sits in
# Điều 4.
INSTRUMENT_RANK: dict[str, int] = {
    "hien_phap": 0,  # Constitution
    "luat": 1,  # Luật (law, passed by Quốc hội)
    "bo_luat": 1,  # Bộ luật (code, e.g. Bộ luật Dân sự) — same tier as luật
    "sac_luat": 1,  # Sắc luật — historical decree-with-force-of-law (Council of Government, DRV transitional period); functionally a luật, not the lower sắc lệnh tier
    "phap_lenh": 2,  # Pháp lệnh (ordinance, UBTVQH)
    "lenh": 2,  # Lệnh (order, President) — same tier as pháp lệnh
    "sac_lenh": 2,  # Sắc lệnh — historical Head-of-State decree; same real-world tier as lệnh / quyết định của Chủ tịch nước
    "nghi_quyet_lien_tich": 2,  # Nghị quyết liên tịch (UBTVQH/Chính phủ jointly with MTTQ) — Điều 4 groups this with pháp lệnh
    "nghi_dinh": 3,  # Nghị định (decree, Government)
    "thong_tu": 4,
    "thong_tu_lien_tich": 4,  # inter-ministerial circular — same tier as thông tư
    "thong_tu_lien_bo": 4,  # historical (pre-2008ish) name for the same inter-ministerial circular
    "quyet_dinh": 5,
    "chi_thi": 5,  # directive — implementing-level, same tier as quyết định/chỉ thị
    "nghi_quyet": 5,  # DEFAULT tier for this slug — see `rank_of()`, this is the dia_phuong (HĐND) case; trung_uong is promoted to _NGHI_QUYET_CENTRAL_RANK
}

# `van_ban_hop_nhat` (consolidated text, ~1.1% of the corpus) is
# deliberately NOT in the map above. Its own doc_type slug carries no
# information about which original instrument it consolidates (a
# nghị định's consolidated text and a luật's consolidated text share
# the same slug), so guessing a tier here would be worse than leaving
# it unranked — consolidation.py's `_HOP_NHAT_CONFIDENCE` is where this
# codebase already accounts for a văn bản hợp nhất's trustworthiness,
# at the layer that actually has the context (the document it points
# FROM) to judge it.

# Anything not in the map above (an unrecognised or unseen doc_type
# slug, or `van_ban_hop_nhat` per the note above) gets this rank:
# pushed BELOW every known tier rather than guessed into one, and
# flagged via `is_ranked()` so coverage.py can lower its
# hierarchy_confidence instead of silently trusting it.
UNRANKED_FALLBACK = 99

# See point 1 in the module docstring: `doc_type=="nghi_quyet"` is used
# in the real data for BOTH a National Assembly resolution and a
# provincial HĐND one, with no other doc_type-level signal to tell
# them apart. `scope` is the one column that can: `scope=="trung_uong"`
# promotes it to (approximately) the luật-adjacent tier a National
# Assembly resolution actually holds under Điều 4. `scope==
# "dia_phuong"`, or a caller that doesn't pass scope at all, uses the
# implementing tier already in INSTRUMENT_RANK above — the safer
# default given dia_phuong is 65.7% of the whole corpus, and strictly
# better than this slug's old behavior of falling to
# UNRANKED_FALLBACK regardless of level.
_NGHI_QUYET_CENTRAL_RANK = 1


def rank_of(doc_type: str, scope: str | None = None) -> int:
    if doc_type == "nghi_quyet" and scope == "trung_uong":
        return _NGHI_QUYET_CENTRAL_RANK
    return INSTRUMENT_RANK.get(doc_type, UNRANKED_FALLBACK)


def is_ranked(doc_type: str, scope: str | None = None) -> bool:
    if doc_type == "nghi_quyet":
        return True  # always resolvable to a tier now, see rank_of()
    return doc_type in INSTRUMENT_RANK


def more_authoritative(
    doc_type_a: str,
    doc_type_b: str,
    *,
    scope_a: str | None = None,
    scope_b: str | None = None,
) -> bool:
    """True if `doc_type_a` outranks (binds) `doc_type_b`. Equal rank
    returns False for both directions — same tier, no precedence
    between them from this map alone. `scope_a`/`scope_b` only affect
    the outcome when the corresponding doc_type is "nghi_quyet" — see
    `rank_of()`."""
    return rank_of(doc_type_a, scope_a) < rank_of(doc_type_b, scope_b)
