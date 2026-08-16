"""Regression test for a real bug found by actually running the app
(not just its test suite): `run_ingestion.ingest_sample_fixture()`
originally tagged its demo document `category="test_fixture"`, but
`app.rag.query_builder.applicable_categories()` only ever returns real
categories from `CATEGORY_TRIGGER_TRAITS`. Following the README's own
Quick Start (ingest sample fixture, then click "Generate Checklist")
therefore always produced a silently EMPTY checklist — no error, just
nothing — which is exactly the wrong first impression for a feature
whose whole point is a live demo.
"""
from __future__ import annotations

from app.ingestion.run_ingestion import ingest_sample_fixture
from app.rag.query_builder import CATEGORY_TRIGGER_TRAITS


def test_sample_fixture_category_is_one_the_checklist_generator_actually_queries(monkeypatch):
    captured_source = {}

    def _fake_ingest_document(doc_id, title, raw_text, source, **kwargs):
        captured_source["source"] = source
        return {"written": 0, "skipped": 0, "total": 0, "document_id": doc_id, "article_count": 0}

    monkeypatch.setattr("app.ingestion.run_ingestion.ingest_document", _fake_ingest_document)

    ingest_sample_fixture()

    category = captured_source["source"].category
    assert category in CATEGORY_TRIGGER_TRAITS, (
        f"ingest_sample_fixture() tagged the demo document category={category!r}, which "
        f"isn't one of {list(CATEGORY_TRIGGER_TRAITS)} — the checklist generator will never "
        "query it, so the Quick Start's manual-generate button will silently produce an "
        "empty checklist for anyone following the README."
    )
