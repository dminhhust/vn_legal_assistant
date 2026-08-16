"""Integration tests for POST /chat — uses dependency overrides for
both the DB session and the LegalChatAgent (injecting a fake router),
so no real LLM call and no real Postgres are needed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chat.agent import LegalChatAgent
from app.chat.router import get_chat_agent
from app.db.session import Base, get_db
from app.llm.schemas import LLMResponse
from app.main import app


class _FakeRouter:
    def __init__(self, text: str):
        self._text = text

    def complete(self, messages, **kwargs):
        return LLMResponse(text=self._text, tool_calls=[], provider="fake", model="fake")


@pytest.fixture()
def client():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        db = session_local()
        try:
            yield db
        finally:
            db.close()

    def override_get_chat_agent():
        return LegalChatAgent(router=_FakeRouter(text="Hello!"))

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_chat_agent] = override_get_chat_agent
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _make_user(client) -> str:
    resp = client.post("/profile", json={"username": "chatuser"})
    return resp.json()["user_id"]


def test_chat_endpoint_returns_text(client):
    user_id = _make_user(client)
    resp = client.post("/chat", json={"user_id": user_id, "message": "hi there"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello!"
    assert data["tool_calls_made"] == []


def test_chat_endpoint_accepts_history(client):
    user_id = _make_user(client)
    app.dependency_overrides[get_chat_agent] = lambda: LegalChatAgent(
        router=_FakeRouter(text="Grounded legal answer.")
    )

    resp = client.post(
        "/chat",
        json={
            "user_id": user_id,
            "message": "what taxes do I owe?",
            "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        },
    )

    assert resp.status_code == 200
    assert resp.json()["text"] == "Grounded legal answer."


def test_chat_endpoint_missing_fields_returns_422(client):
    resp = client.post("/chat", json={"message": "hi"})  # missing user_id
    assert resp.status_code == 422


def test_chat_endpoint_no_llm_provider_returns_clean_503_not_bare_500(client):
    """Regression test: running the real server with no LLM key
    configured produced an unhandled NoProviderAvailableError -> a bare
    500 with no useful body. Same class of bug as the checklist
    endpoint's equivalent test — only caught by actually running the
    live server without a key set, since every other test's fixture
    always supplies a working fake router."""
    from app.llm.router import NoProviderAvailableError

    class _NoProviderRouter:
        def complete(self, messages, **kwargs):
            raise NoProviderAvailableError("No available provider for task='chat'. Available providers: none.")

    user_id = _make_user(client)
    app.dependency_overrides[get_chat_agent] = lambda: LegalChatAgent(router=_NoProviderRouter())

    resp = client.post("/chat", json={"user_id": user_id, "message": "hi"})

    assert resp.status_code == 503
    assert "LLM provider" in resp.json()["detail"]


def test_chat_endpoint_no_llm_provider_returns_clean_503_not_bare_500(client):
    """Regression test — mirrors
    test_checklist_api.py::test_generate_checklist_no_llm_provider_returns_clean_503_not_bare_500.
    Running the live server with no LLM key configured produced an
    unhandled NoProviderAvailableError -> a bare 500 here too, since
    LegalChatAgent.handle()'s first router.complete() call was never
    wrapped. Caught only by actually running the server without a key,
    not by this test suite's fake-router fixtures, which always
    provided a working fake."""
    from app.llm.router import NoProviderAvailableError

    class _NoProviderRouter:
        def complete(self, messages, **kwargs):
            raise NoProviderAvailableError("No available provider for task='chat'. Available providers: none.")

    app.dependency_overrides[get_chat_agent] = lambda: LegalChatAgent(router=_NoProviderRouter())
    user_id = _make_user(client)

    resp = client.post("/chat", json={"user_id": user_id, "message": "hi"})

    assert resp.status_code == 503
    assert "LLM provider" in resp.json()["detail"]
