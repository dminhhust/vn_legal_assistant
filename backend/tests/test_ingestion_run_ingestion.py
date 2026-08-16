"""Regression test for a real bug found by actually running the app:
`ingest_sample_fixture()` originally tagged its demo data with
category="test_fixture", which app.rag.query_builder's
CATEGORY_TRIGGER_TRAITS never queries (it only ever asks for the 7 real
categories). Following the README's own Quick Start — ingest the
sample fixture, then click "Generate My Checklist" — produced a
silently empty checklist with zero explanation, which is exactly the
wrong first impression for a feature whose entire point is a live demo.

This wasn't caught by test_checklist_service.py /
test_checklist_api.py because those tests ingest their OWN fixture data
directly via ingest_document(..., category="tax", ...) — they never go
through ingest_sample_fixture() at all, so a wrong category tag there
was invisible to them. Only caught by running the actual CLI against a
live Chroma server and then hitting the real /checklist/.../generate
endpoint and seeing an unexplained empty list.
"""
from __future__ import annotations

from app.ingestion.run_ingestion import ingest_sample_fixture
from app.rag.query_builder import CATEGORY_TRIGGER_TRAITS


def test_sample_fixture_is_tagged_with_a_real_checklist_category(monkeypatch):
    captured_source = {}

    def fake_ingest_document(doc_id, title, raw_text, source, **kwargs):
        captured_source["source"] = source
        return {"written": 0, "skipped": 0, "total": 0, "document_id": doc_id, "article_count": 0}

    monkeypatch.setattr("app.ingestion.run_ingestion.ingest_document", fake_ingest_document)

    ingest_sample_fixture()

    category = captured_source["source"].category
    assert category in CATEGORY_TRIGGER_TRAITS, (
        f"ingest_sample_fixture() tagged its demo data with category={category!r}, "
        f"which isn't one of the real categories {list(CATEGORY_TRIGGER_TRAITS)} that "
        "app.rag.checklist_service actually queries. A fresh install following the "
        "README's Quick Start (ingest sample data, then click 'Generate My Checklist') "
        "would silently get an empty checklist with no explanation."
    )
