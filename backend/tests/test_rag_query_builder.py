"""Unit tests for query_builder.py."""
from __future__ import annotations

from app.rag.query_builder import applicable_categories, build_category_queries, infer_entity_type


def test_baseline_categories_always_applicable_with_no_traits():
    categories = applicable_categories([])
    assert "tax" in categories
    assert "labor_insurance" in categories
    assert "contracts_signing" in categories
    assert "residence_civil" in categories


def test_business_licensing_only_applicable_for_business_owner():
    assert "business_licensing" not in applicable_categories([])
    assert "business_licensing" in applicable_categories(["small_business_owner"])


def test_property_vehicles_applicable_if_either_trait_present():
    assert "property_vehicles" in applicable_categories(["property_owner"])
    assert "property_vehicles" in applicable_categories(["vehicle_owner"])
    assert "property_vehicles" not in applicable_categories(["freelancer"])


def test_family_civil_applicable_for_married_or_dependents():
    assert "family_civil" in applicable_categories(["marital_married"])
    assert "family_civil" in applicable_categories(["has_dependents"])
    assert "family_civil" not in applicable_categories(["marital_single"])


def test_build_category_queries_returns_one_query_per_applicable_category():
    queries = build_category_queries(["small_business_owner"])
    categories = {q.category for q in queries}
    assert "business_licensing" in categories
    assert "tax" in categories  # baseline
    # every returned query has non-empty text
    assert all(q.query_text for q in queries)


def test_matched_traits_recorded_for_triggered_categories():
    queries = build_category_queries(["small_business_owner"])
    biz_query = next(q for q in queries if q.category == "business_licensing")
    assert "small_business_owner" in biz_query.matched_traits


def test_baseline_category_has_empty_matched_traits():
    queries = build_category_queries(["small_business_owner"])
    tax_query = next(q for q in queries if q.category == "tax")
    assert tax_query.matched_traits == []


def test_infer_entity_type_business_owner():
    assert infer_entity_type(["small_business_owner"]) == "business"


def test_infer_entity_type_individual_default():
    assert infer_entity_type([]) == "individual"
    assert infer_entity_type(["freelancer", "has_dependents"]) == "individual"
