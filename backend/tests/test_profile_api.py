"""Integration tests for the Profile Service API.

Uses an in-memory SQLite DB (via a `get_db` dependency override) so
these tests never touch a real Postgres instance and never need
docker-compose running. Each test gets a fresh engine/schema via the
`client` fixture, so tests can't leak state into each other.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.session import Base, get_db
from app.main import app


@pytest.fixture()
def client():
    # StaticPool is required here: FastAPI's TestClient runs sync route
    # handlers in a worker thread. Without StaticPool, SQLite's default
    # pooling hands out a NEW connection per thread, and each connection
    # to ":memory:" is its own separate, empty database — the schema
    # created below would be invisible to the actual request handling.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _onboard_payload(username: str = "testuser", **overrides) -> dict:
    payload = {
        "username": username,
        "age": 30,
        "occupation_type": "business_owner",
        "has_business": True,
        "business_sector": "retail",
        "dependents": 1,
        "owns_property": True,
    }
    payload.update(overrides)
    return payload


def test_onboarding_creates_profile_with_derived_traits(client):
    resp = client.post("/profile", json=_onboard_payload())
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "testuser"
    assert data["version"] == 1
    assert "small_business_owner" in data["traits"]
    assert "business_sector_retail" in data["traits"]
    assert "has_dependents" in data["traits"]
    assert "property_owner" in data["traits"]


def test_onboarding_with_only_required_field_uses_sensible_defaults(client):
    resp = client.post("/profile", json={"username": "minimal_user"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["dependents"] == 0
    assert data["has_business"] is False
    assert data["traits"] == []


def test_duplicate_username_rejected(client):
    client.post("/profile", json=_onboard_payload())
    resp = client.post("/profile", json=_onboard_payload())
    assert resp.status_code == 409


def test_get_profile_returns_stored_data(client):
    created = client.post("/profile", json=_onboard_payload()).json()
    resp = client.get(f"/profile/{created['user_id']}")
    assert resp.status_code == 200
    assert resp.json()["username"] == "testuser"


def test_get_nonexistent_profile_returns_404(client):
    resp = client.get("/profile/does-not-exist")
    assert resp.status_code == 404


def test_editing_profile_increments_version_and_updates_traits(client):
    created = client.post("/profile", json=_onboard_payload()).json()
    user_id = created["user_id"]
    assert created["version"] == 1

    resp = client.put(f"/profile/{user_id}", json={"owns_vehicle": True})
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["version"] == 2
    assert "vehicle_owner" in updated["traits"]
    assert "property_owner" in updated["traits"]  # untouched, still present


def test_editing_unrelated_field_does_not_change_other_traits(client):
    created = client.post("/profile", json=_onboard_payload()).json()
    user_id = created["user_id"]

    resp = client.put(f"/profile/{user_id}", json={"age": 31})
    updated = resp.json()
    assert updated["age"] == 31
    assert "small_business_owner" in updated["traits"]  # untouched by the unrelated edit


def test_no_op_update_does_not_bump_version(client):
    created = client.post("/profile", json=_onboard_payload()).json()
    user_id = created["user_id"]

    # Re-sending the exact same value that's already stored is a no-op.
    resp = client.put(f"/profile/{user_id}", json={"age": created["age"]})
    assert resp.json()["version"] == created["version"]


def test_edit_nonexistent_profile_returns_404(client):
    resp = client.put("/profile/does-not-exist", json={"age": 40})
    assert resp.status_code == 404
