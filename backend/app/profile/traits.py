"""Trait derivation rules layer.

Converts raw profile fields into semantic trait tags that drive legal-
obligation filtering (docs/ARCHITECTURE.md "Legal RAG pipeline").

Deliberately implemented as small, independent, pure functions rather
than one big branching function — adding a new rule means adding a new
function and registering it in `_RULES`; it never requires editing an
existing rule. Each rule takes a plain dict of profile field values (see
schemas.PROFILE_FIELD_NAMES) and returns zero or more tag strings.
"""
from __future__ import annotations

from typing import Callable

ProfileData = dict
RuleFn = Callable[[ProfileData], list[str]]


def _rule_occupation(p: ProfileData) -> list[str]:
    occ = p.get("occupation_type")
    return [occ] if occ else []


def _rule_business_owner(p: ProfileData) -> list[str]:
    if not p.get("has_business"):
        return []
    tags = ["small_business_owner"]
    sector = p.get("business_sector")
    if sector:
        tags.append(f"business_sector_{sector}")
    return tags


def _rule_dependents(p: ProfileData) -> list[str]:
    return ["has_dependents"] if (p.get("dependents") or 0) > 0 else []


def _rule_marital_status(p: ProfileData) -> list[str]:
    status = p.get("marital_status")
    return [f"marital_{status}"] if status else []


def _rule_owns_property(p: ProfileData) -> list[str]:
    return ["property_owner"] if p.get("owns_property") else []


def _rule_owns_vehicle(p: ProfileData) -> list[str]:
    return ["vehicle_owner"] if p.get("owns_vehicle") else []


def _rule_province(p: ProfileData) -> list[str]:
    # Drives residence/local-scope obligation filtering (metadata
    # filtering: national vs. local-scope legal provisions).
    province = p.get("province")
    return [f"resident_of_{province}"] if province else []


# Registry: adding a rule here is the ONLY change needed to extend trait
# derivation — no other file needs to change.
_RULES: list[RuleFn] = [
    _rule_occupation,
    _rule_business_owner,
    _rule_dependents,
    _rule_marital_status,
    _rule_owns_property,
    _rule_owns_vehicle,
    _rule_province,
]


def derive_traits(profile: ProfileData) -> list[tuple[str, str]]:
    """Runs every rule against the profile and unions the results.

    Returns a list of (tag, source_rule_name) pairs, de-duplicated by
    tag — if two rules ever produced the same tag, the first one in
    `_RULES` gets credit, which only matters for the debugging label.
    """
    seen: dict[str, str] = {}
    for rule in _RULES:
        for tag in rule(profile):
            seen.setdefault(tag, rule.__name__)
    return list(seen.items())
