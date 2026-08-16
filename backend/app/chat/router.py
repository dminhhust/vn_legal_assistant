"""Chat endpoint — the RAG chatbot feature. POST /chat runs the
message through LegalChatAgent (app/chat/agent.py), which decides
whether to call the legal-search or checklist tools before answering.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.chat.agent import LegalChatAgent
from app.db.session import get_db
from app.llm.router import NoProviderAvailableError

router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_agent() -> LegalChatAgent:
    """FastAPI dependency so tests can override this to inject a
    LegalChatAgent built with a fake router — same DI-override pattern
    as `get_db` — without needing a real LLM provider key or a running
    Chroma instance."""
    return LegalChatAgent()


class ChatTurn(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    user_id: str
    message: str
    # Optional prior turns, oldest first — the client resends these
    # each request so the agent has conversational context without a
    # server-side session store (see chat/agent.py's module docstring).
    history: list[ChatTurn] = []


class ChatResponse(BaseModel):
    text: str
    tool_calls_made: list[str]


@router.post("", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    agent: LegalChatAgent = Depends(get_chat_agent),
) -> ChatResponse:
    history = [(turn.role, turn.content) for turn in payload.history]
    try:
        result = agent.handle(payload.message, user_id=payload.user_id, db=db, history=history)
    except NoProviderAvailableError as exc:
        # Same reasoning as app/rag/router.py's generate_checklist: a
        # missing LLM key is routine here, not a bug, and this endpoint
        # is meant to be demoed live — surface it as a clear 503, not a
        # bare 500. Found the same way — running the real server with
        # no key configured, which the test suite's fake-router
        # fixtures never do.
        raise HTTPException(
            status_code=503,
            detail=(
                "Chat needs at least one working LLM provider. "
                f"{exc} Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY "
                "in backend/.env and restart the backend."
            ),
        ) from exc
    return ChatResponse(text=result.text, tool_calls_made=result.tool_calls_made)
