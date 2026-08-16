"""Integration tests for /checklist endpoints (app/rag/router.py) —
real SQLite DB, real in-memory Chroma with a synthetic legal fixture
ingested, fake LLM router (no real API call).
"""
from __future__ import annotations

import uuid

import chromadb
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.ingestion.embeddings import HashingEmbeddingProvider
from app.ingestion.metadata import SourceMeta
from app.ingestion.pipeline import ingest_document
from app.ingestion.vector_store import VectorStoreWriter
from app.llm.schemas import LLMResponse
from app.main import app
from app.rag.router import get_checklist_generator_kwargs

FIXTURE_TEXT = "Điều 1. Test obligation\n1. Do the test thing by March 31 each year.\n"


class _FakeExtractionRouter:
    def complete(self, messages, **kwargs):
        return LLMResponse(
            text=None,
            structured_output={
                "obligations": [
                    {
                        "title": "Fake obligation",
                        "description": "A fake extracted obligation.",
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
            provider="claude",
            model="claude-sonnet-5",
        )


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    vector_store = VectorStoreWriter(client=chromadb.EphemeralClient(), collection_name=f"test-{uuid.uuid4().hex}")
    embedder = HashingEmbeddingProvider()
    ingest_document(
        "test-doc", "Test Doc", FIXTURE_TEXT,
        SourceMeta(law_name="Test Law", category="tax", entity_type="both"),
        vector_store=vector_store, embedder=embedder,
    )

    def override_get_db():
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    def override_gen_kwargs():
        return {"vector_store": vector_store, "embedder": embedder, "llm_router": _FakeExtractionRouter()}

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_checklist_generator_kwargs] = override_gen_kwargs
    with TestClient(app) as test_client:
        # Stashed on the client so individual tests can override JUST the
        # llm_router while still pointing at the same real ingested data
        # (see test_generate_checklist_no_llm_provider_returns_clean_503_not_bare_500) —
        # swapping in a totally fresh default vector_store/embedder for that
        # test would mean retrieval finds nothing and the LLM router is never
        # even called, which would silently defeat the regression test.
        test_client._vector_store = vector_store
        test_client._embedder = embedder
        yield test_client
    app.dependency_overrides.clear()


def _make_user(client) -> str:
    resp = client.post("/profile", json={"username": "checklistuser"})
    return resp.json()["user_id"]


def test_get_checklist_before_generation_is_empty(client):
    user_id = _make_user(client)
    resp = client.get(f"/checklist/{user_id}")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_checklist_unknown_user_returns_404(client):
    resp = client.get("/checklist/does-not-exist")
    assert resp.status_code == 404


def test_generate_checklist_creates_items(client):
    """This is the manual-activation endpoint — the 'showcase' trigger.
    Confirms POSTing to it (with nothing else running in the
    background) is enough to produce a fresh checklist synchronously."""
    user_id = _make_user(client)
    resp = client.post(f"/checklist/{user_id}/generate")

    assert resp.status_code == 200
    items = resp.json()
    assert len(items) >= 1
    assert items[0]["title"] == "Fake obligation"
    assert items[0]["due_date"] == "2026-03-31" or items[0]["due_date"].endswith("-03-31")


def test_generate_checklist_unknown_user_returns_404(client):
    resp = client.post("/checklist/does-not-exist/generate")
    assert resp.status_code == 404


def test_generate_checklist_no_llm_provider_returns_clean_503_not_bare_500(client):
    """Regression test: running the real server with no LLM key
    configured produced an unhandled NoProviderAvailableError -> a bare
    500 with no useful body, on a route whose whole point is to be
    demoed live. The test suite's fixtures always injected a working
    fake LLM router, so this path was never exercised in-process — only
    caught by actually running the live server without a key set.

    Reuses the fixture's already-ingested real data (via the vector_store/
    embedder stashed on `client`) so retrieval genuinely finds a hit and
    the LLM router actually gets called and can raise — swapping in a
    fresh, empty default vector_store here would make retrieval find
    nothing, meaning the router is never invoked and this test would
    pass even without the fix."""
    from app.llm.router import NoProviderAvailableError

    class _NoProviderExtractionRouter:
        def complete(self, messages, **kwargs):
            raise NoProviderAvailableError(
                "No available provider for task='legal_extraction'. Available providers: none."
            )

    def override_gen_kwargs_no_provider():
        return {
            "vector_store": client._vector_store,
            "embedder": client._embedder,
            "llm_router": _NoProviderExtractionRouter(),
        }

    app.dependency_overrides[get_checklist_generator_kwargs] = override_gen_kwargs_no_provider
    user_id = _make_user(client)

    resp = client.post(f"/checklist/{user_id}/generate")

    assert resp.status_code == 503
    assert "LLM provider" in resp.json()["detail"]


def test_get_checklist_after_generation_returns_stored_items(client):
    user_id = _make_user(client)
    client.post(f"/checklist/{user_id}/generate")

    resp = client.get(f"/checklist/{user_id}")
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_update_checklist_item_status(client):
    user_id = _make_user(client)
    generated = client.post(f"/checklist/{user_id}/generate").json()
    item_id = generated[0]["id"]

    resp = client.patch(f"/checklist/{user_id}/items/{item_id}", json={"status": "done"})

    assert resp.status_code == 200
    assert resp.json()["status"] == "done"


def test_update_checklist_item_invalid_status_returns_422(client):
    user_id = _make_user(client)
    generated = client.post(f"/checklist/{user_id}/generate").json()
    item_id = generated[0]["id"]

    resp = client.patch(f"/checklist/{user_id}/items/{item_id}", json={"status": "not_a_real_status"})

    assert resp.status_code == 422


def test_update_nonexistent_item_returns_404(client):
    user_id = _make_user(client)
    resp = client.patch(f"/checklist/{user_id}/items/does-not-exist", json={"status": "done"})
    assert resp.status_code == 404
