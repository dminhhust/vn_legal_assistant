"""Category-scoped query construction (docs/ARCHITECTURE.md §4.2).

Turns a user's derived trait tags (app/profile/traits.py) into one
targeted retrieval query per applicable obligation category, rather
than a single vague "what laws apply to me" query — legal retrieval
needs precision, and category-scoped queries retrieve far more
relevant chunks than one broad one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Maps each obligation category to the trait tags that make it
# applicable to a given user. `None` means "baseline" — applies to
# everyone regardless of traits (e.g. every adult has some tax and
# residence obligations). This is config data, not branching code, so
# adding a new category's applicability rule is a one-line change here,
# matching the same principle used for the category taxonomy itself
# (app/ingestion/metadata.py) and the LLM task-priority policy
# (app/llm/task_policy.py).
CATEGORY_TRIGGER_TRAITS: dict[str, Optional[list[str]]] = {
    "tax": None,
    "labor_insurance": None,
    "contracts_signing": None,
    "residence_civil": None,
    "business_licensing": ["small_business_owner"],
    "property_vehicles": ["property_owner", "vehicle_owner"],
    "family_civil": ["marital_married", "has_dependents"],
}

# Human-readable category descriptions used to phrase a retrieval query
# — kept separate from the taxonomy itself (app/ingestion/metadata.py)
# so this file owns "how to phrase a query for this category" without
# duplicating the category list.
#
# IMPORTANT — Vietnamese on purpose. The corpus is Vietnamese legal
# text, and BM25 (`app/rag/retrieval.py`) is an exact-term signal: an
# English template ("tax obligations, filing deadlines") shares ~zero
# tokens with Vietnamese chunks, so lexical scoring contributed nothing
# and ranking collapsed to corpus order (found by actually measuring
# retrieval per category against the real corpus — see
# docs/PERFORMANCE_ANALYSIS.md §4.2). Vietnamese templates restore the
# BM25 signal that this codebase explicitly treats as its degradation-
# safe baseline.
_CATEGORY_QUERY_TEMPLATES: dict[str, str] = {
    "tax": "nghĩa vụ thuế của cá nhân và doanh nghiệp, thời hạn kê khai và nộp thuế",
    "labor_insurance": "nghĩa vụ của người lao động và người sử dụng lao động về lao động, bảo hiểm xã hội và bảo hiểm y tế",
    "contracts_signing": "nghĩa vụ ký kết hợp đồng, công chứng và đăng ký giao dịch",
    "residence_civil": "nghĩa vụ đăng ký cư trú, hộ tịch và tình trạng dân sự",
    "business_licensing": "nghĩa vụ đăng ký doanh nghiệp, giấy phép kinh doanh và các điều kiện hoạt động",
    "property_vehicles": "nghĩa vụ đăng ký tài sản, nhà đất và phương tiện giao thông",
    "family_civil": "nghĩa vụ gia đình và dân sự như kết hôn, nuôi con và thừa kế",
}


@dataclass
class CategoryQuery:
    category: str
    query_text: str
    matched_traits: list[str]  # which of the user's traits triggered this category (empty for baseline)


def infer_entity_type(trait_tags: list[str]) -> str:
    """Coarse individual-vs-business classification used to filter
    retrieval candidates by their `entity_type` metadata
    (app/ingestion/metadata.py) — a chunk tagged strictly "business"
    shouldn't surface for a user with no business, and vice versa."""
    return "business" if "small_business_owner" in trait_tags else "individual"


def applicable_categories(trait_tags: list[str]) -> list[str]:
    """Every category considered applicable to a user with the given
    trait tags — baseline categories always included, others only if a
    triggering trait is present."""
    trait_set = set(trait_tags)
    return [
        category
        for category, triggers in CATEGORY_TRIGGER_TRAITS.items()
        if triggers is None or trait_set.intersection(triggers)
    ]


def build_category_queries(trait_tags: list[str]) -> list[CategoryQuery]:
    """One query per applicable category — never one combined query for
    everything, since that flattens retrieval precision (§4.2). Query
    text is Vietnamese (see `_CATEGORY_QUERY_TEMPLATES`) and is
    personalized with the user's triggered traits AND their business
    sector when present (e.g. "đối với small_business_owner,
    business_sector_technology") so a restaurant and a software house
    don't get byte-identical queries for the same category."""
    trait_set = set(trait_tags)
    # business_sector_* traits are derived but don't trigger any
    # category on their own — they only refine the query text.
    sector_tags = sorted(t for t in trait_set if t.startswith("business_sector_"))
    queries = []
    for category in applicable_categories(trait_tags):
        triggers = CATEGORY_TRIGGER_TRAITS[category]
        matched = sorted(trait_set.intersection(triggers)) if triggers else []
        base_query = _CATEGORY_QUERY_TEMPLATES.get(category, category)
        context_tags = matched + sector_tags
        query_text = (
            f"{base_query} đối với {', '.join(context_tags)}" if context_tags else base_query
        )
        queries.append(CategoryQuery(category=category, query_text=query_text, matched_traits=matched))
    return queries
