"""Unit tests for LegalChatAgent — uses a fake router with scripted
sequential responses (first call: tool selection; second call: final
synthesis), never a real LLM API call."""
from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chat.agent import LegalChatAgent
from app.db.models import ObligationChecklistItem
from app.db.session import Base
from app.llm.schemas import LLMResponse, ToolCall
from app.profile.schemas import ProfileIn
from app.profile.service import create_profile


class _FakeSequentialRouter:
    """Returns each entry in `responses` in order, one per call — lets a
    test script a full "tool selection -> tool execution -> final
    synthesis" round trip without a real LLM."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self._responses.pop(0)


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
def user_profile(db_session):
    return create_profile(db_session, ProfileIn(username="agentuser"))


def test_handle_with_no_tool_call_returns_direct_text(db_session, user_profile):
    fake_router = _FakeSequentialRouter(
        [LLMResponse(text="Hello! How can I help?", tool_calls=[], provider="fake", model="fake")]
    )
    agent = LegalChatAgent(router=fake_router)

    result = agent.handle("hi", user_id=user_profile.user_id, db=db_session)

    assert result.text == "Hello! How can I help?"
    assert result.tool_calls_made == []
    assert len(fake_router.calls) == 1  # no follow-up call needed


def test_handle_executes_checklist_status_tool_and_synthesizes_answer(db_session, user_profile):
    item = ObligationChecklistItem(
        user_id=user_profile.user_id,
        title="File taxes",
        category="tax",
        description="d",
        deadline_type="fixed",
        due_date=date(2026, 3, 31),
        penalty_summary="p",
        source_citation="Test Law, Điều 1",
        source_chunk_id="x",
        status="pending",
    )
    db_session.add(item)
    db_session.commit()

    fake_router = _FakeSequentialRouter(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="call1", name="get_checklist_status", arguments={"days_ahead": None})],
                provider="fake",
                model="fake",
            ),
            LLMResponse(
                text="You have one pending item: File taxes, due 2026-03-31.",
                tool_calls=[],
                provider="fake",
                model="fake",
            ),
        ]
    )
    agent = LegalChatAgent(router=fake_router)

    result = agent.handle("what's due?", user_id=user_profile.user_id, db=db_session)

    assert "File taxes" in result.text
    assert result.tool_calls_made == ["get_checklist_status"]
    assert len(fake_router.calls) == 2


def test_handle_executes_mark_done_tool(db_session, user_profile):
    item = ObligationChecklistItem(
        user_id=user_profile.user_id,
        title="Renew license",
        category="business_licensing",
        description="d",
        deadline_type="fixed",
        due_date=date(2026, 6, 1),
        penalty_summary="p",
        source_citation="Test Law, Điều 2",
        source_chunk_id="y",
        status="pending",
    )
    db_session.add(item)
    db_session.commit()

    fake_router = _FakeSequentialRouter(
        [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="call1", name="mark_checklist_item_done", arguments={"title_contains": "license"})
                ],
                provider="fake",
                model="fake",
            ),
            LLMResponse(text="Done! I've marked it as complete.", tool_calls=[], provider="fake", model="fake"),
        ]
    )
    agent = LegalChatAgent(router=fake_router)

    result = agent.handle("mark the license renewal as done", user_id=user_profile.user_id, db=db_session)

    db_session.refresh(item)
    assert item.status == "done"
    assert result.tool_calls_made == ["mark_checklist_item_done"]


def test_handle_unknown_tool_name_does_not_crash(db_session, user_profile):
    fake_router = _FakeSequentialRouter(
        [
            LLMResponse(
                text=None,
                tool_calls=[ToolCall(id="call1", name="some_future_tool", arguments={})],
                provider="fake",
                model="fake",
            ),
            LLMResponse(text="I couldn't find a matching tool.", tool_calls=[], provider="fake", model="fake"),
        ]
    )
    agent = LegalChatAgent(router=fake_router)

    result = agent.handle("do something unsupported", user_id=user_profile.user_id, db=db_session)

    assert result.tool_calls_made == ["some_future_tool"]
    assert "couldn't find" in result.text.lower()


def test_handle_executes_multiple_tool_calls_in_one_turn(db_session, user_profile):
    item = ObligationChecklistItem(
        user_id=user_profile.user_id,
        title="File taxes",
        category="tax",
        description="d",
        deadline_type="fixed",
        due_date=date(2026, 3, 31),
        penalty_summary="p",
        source_citation="Test Law, Điều 1",
        source_chunk_id="x",
        status="pending",
    )
    db_session.add(item)
    db_session.commit()

    fake_router = _FakeSequentialRouter(
        [
            LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="c1", name="get_checklist_status", arguments={"days_ahead": None}),
                    ToolCall(id="c2", name="mark_checklist_item_done", arguments={"title_contains": "taxes"}),
                ],
                provider="fake",
                model="fake",
            ),
            LLMResponse(text="Here's your status and I marked it done.", tool_calls=[], provider="fake", model="fake"),
        ]
    )
    agent = LegalChatAgent(router=fake_router)

    result = agent.handle("what's due, and mark taxes done", user_id=user_profile.user_id, db=db_session)

    assert result.tool_calls_made == ["get_checklist_status", "mark_checklist_item_done"]
    db_session.refresh(item)
    assert item.status == "done"
