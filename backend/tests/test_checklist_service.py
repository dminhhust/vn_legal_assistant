"""Integration test for the full Phase 3 pipeline: profile -> traits ->
category queries -> hybrid retrieval -> extraction -> due-date
computation -> persisted checklist.

Uses:
  - a real SQLite DB (same StaticPool pattern as test_profile_api.py)
  - a real in-memory Chroma (EphemeralClient) with the synthetic Phase 2
    fixture ingested
  - the offline HashingEmbeddingProvider (retrieval PLUMBING is what's
    under test — see app/ingestion/embeddings.py's docstring)
  - a FAKE LLM router with a scripted response — no real API call
This is the closest thing this codebase has to the Phase 3 Definition
of Done from docs/IMPLEMENTATION_PLAN.md, run as an automated test
instead of only a manual script.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
import uuid

import chromadb
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import ObligationChecklistItem
from app.db.session import Base
from app.ingestion.embeddings import HashingEmbeddingProvider
from app.ingestion.metadata import SourceMeta
from app.ingestion.pipeline import ingest_document
from app.ingestion.vector_store import VectorStoreWriter
from app.llm.schemas import LLMResponse
from app.profile.schemas import ProfileIn
from app.profile.service import create_profile
from app.rag.checklist_service import UserNotFoundError, generate_checklist_for_user

FIXTURE_PATH = (
    Path(__file__).parent.parent / "app" / "ingestion" / "sample_data" / "sample_test_law.txt"
)


class _FakeRouter:
    """Returns one scripted obligation per call regardless of prompt —
    enough to verify the extraction -> due-date -> persistence wiring
    without a real LLM call."""

    def __init__(self):
        self.call_count = 0

    def complete(self, messages, **kwargs):
        self.call_count += 1
        return LLMResponse(
            text=None,
            structured_output={
                "obligations": [
                    {
                        "title": f"Fake obligation #{self.call_count}",
                        "description": "A fake obligation extracted for testing.",
                        "deadline_type": "fixed",
                        "deadline_month": 3,
                        "deadline_day": 31,
                        "period_months": None,
                        "days_after_event": None,
                        "event_description": None,
                        "penalty_summary": "A fake penalty.",
                    }
                ]
            },
            provider="fake",
            model="fake-model",
        )


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_local()
    yield session
    session.close()


@pytest.fixture()
def vector_store():
    return VectorStoreWriter(client=chromadb.EphemeralClient(), collection_name=f"test-{uuid.uuid4().hex}")


@pytest.fixture()
def embedder():
    return HashingEmbeddingProvider()


def _make_business_owner_profile(db_session):
    payload = ProfileIn(username="bizowner", has_business=True, business_sector="retail", province="Hanoi")
    return create_profile(db_session, payload)


def _ingest_fixture_as_tax(vector_store, embedder):
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    source = SourceMeta(
        law_name="SAMPLE_TEST_LAW (synthetic, NOT real law) — tagged as tax for pipeline testing",
        category="tax",
        entity_type="both",
    )
    return ingest_document(
        "sample-test-law", "Sample Test Law", text, source, vector_store=vector_store, embedder=embedder
    )


def test_checklist_generation_end_to_end(db_session, vector_store, embedder):
    profile = _make_business_owner_profile(db_session)
    _ingest_fixture_as_tax(vector_store, embedder)

    fake_router = _FakeRouter()
    saved_items = generate_checklist_for_user(
        db_session,
        profile.user_id,
        vector_store=vector_store,
        embedder=embedder,
        llm_router=fake_router,
        today=date(2026, 1, 1),
    )

    assert len(saved_items) > 0
    assert fake_router.call_count > 0

    first = saved_items[0]
    assert first.category == "tax"
    assert first.due_date == date(2026, 3, 31)
    assert "SAMPLE_TEST_LAW" in first.source_citation
    assert first.status == "pending"


def test_business_owner_gets_business_licensing_obligations_too(db_session, vector_store, embedder):
    _ingest_fixture_as_tax(vector_store, embedder)
    # Also ingest a business-licensing-tagged doc so we can confirm a
    # business owner's checklist actually spans multiple categories.
    ingest_document(
        "biz-doc",
        "Business Doc",
        "Điều 1. Business registration\n1. Register your business within 30 days of starting operations.\n",
        SourceMeta(law_name="Biz Law (synthetic)", category="business_licensing", entity_type="business"),
        vector_store=vector_store,
        embedder=embedder,
    )
    profile = _make_business_owner_profile(db_session)

    saved_items = generate_checklist_for_user(
        db_session, profile.user_id, vector_store=vector_store, embedder=embedder, llm_router=_FakeRouter()
    )

    categories = {item.category for item in saved_items}
    assert "tax" in categories
    assert "business_licensing" in categories


def test_regenerating_checklist_replaces_previous_items_wholesale(db_session, vector_store, embedder):
    profile = _make_business_owner_profile(db_session)
    _ingest_fixture_as_tax(vector_store, embedder)

    generate_checklist_for_user(
        db_session, profile.user_id, vector_store=vector_store, embedder=embedder, llm_router=_FakeRouter()
    )
    first_run_count = (
        db_session.query(ObligationChecklistItem).filter_by(user_id=profile.user_id).count()
    )

    generate_checklist_for_user(
        db_session, profile.user_id, vector_store=vector_store, embedder=embedder, llm_router=_FakeRouter()
    )
    second_run_count = (
        db_session.query(ObligationChecklistItem).filter_by(user_id=profile.user_id).count()
    )

    # Wholesale regeneration: count shouldn't double, confirming old rows were cleared first.
    assert second_run_count == first_run_count
    assert second_run_count > 0


def test_unknown_user_raises(db_session, vector_store, embedder):
    with pytest.raises(UserNotFoundError):
        generate_checklist_for_user(
            db_session, "does-not-exist", vector_store=vector_store, embedder=embedder, llm_router=_FakeRouter()
        )


def test_user_with_no_applicable_extra_categories_still_gets_baseline(db_session, vector_store, embedder):
    # A minimal profile (no business, no property, unmarried, no
    # dependents) still has baseline categories (tax, labor_insurance,
    # contracts_signing, residence_civil) — confirms the checklist
    # generator doesn't silently produce nothing for a "plain" user.
    payload = ProfileIn(username="plain_user")
    profile = create_profile(db_session, payload)
    _ingest_fixture_as_tax(vector_store, embedder)

    saved_items = generate_checklist_for_user(
        db_session, profile.user_id, vector_store=vector_store, embedder=embedder, llm_router=_FakeRouter()
    )

    assert len(saved_items) > 0
    assert all(item.category == "tax" for item in saved_items)  # only tax-tagged data exists in this test
