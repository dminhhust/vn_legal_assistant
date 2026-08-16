"""Checklist HTTP endpoints.

`POST /checklist/{user_id}/generate` is the MANUAL ACTIVATION MECHANISM
for the auto-generated checklist feature: nothing runs the legal RAG +
extraction pipeline automatically (no scheduler, no "runs on
onboarding" hook) — a person (the end user, or whoever is presenting a
demo) explicitly triggers it, and the response is exactly the freshly
generated checklist. That's deliberate for an MVP/showcase: the whole
retrieval -> extraction -> deadline-computation pipeline runs live,
in front of whoever clicked the button, rather than happening silently
in the background where it can't be demonstrated. See
docs/ARCHITECTURE.md "Auto-generated checklist with a manual activation
mechanism".
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import ObligationChecklistItem, User
from app.db.session import get_db
from app.llm.router import NoProviderAvailableError
from app.rag import checklist_service

router = APIRouter(prefix="/checklist", tags=["checklist"])


def get_checklist_generator_kwargs() -> dict:
    """FastAPI dependency so tests can override this to inject a fake
    vector_store/embedder/llm_router into generate_checklist_for_user.
    Empty dict means 'use the real infra defaults baked into
    generate_checklist_for_user itself'."""
    return {}


class ChecklistItemOut(BaseModel):
    id: str
    title: str
    category: str
    description: str
    due_date: Optional[str] = None
    penalty_summary: str
    source_citation: str
    status: str


class ChecklistItemStatusUpdate(BaseModel):
    status: str  # "pending" | "done" | "dismissed" — kept as plain str, validated below


_VALID_STATUSES = {"pending", "done", "dismissed"}


def _to_item_out(item: ObligationChecklistItem) -> ChecklistItemOut:
    return ChecklistItemOut(
        id=item.id,
        title=item.title,
        category=item.category,
        description=item.description,
        due_date=str(item.due_date) if item.due_date else None,
        penalty_summary=item.penalty_summary,
        source_citation=item.source_citation,
        status=item.status,
    )


@router.get("/{user_id}", response_model=list[ChecklistItemOut])
def get_checklist(user_id: str, db: Session = Depends(get_db)) -> list[ChecklistItemOut]:
    """Returns the CURRENTLY STORED checklist — does not regenerate.
    Empty list (not an error) for a user who exists but hasn't
    generated a checklist yet; 404 only for a genuinely unknown user."""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    items = db.query(ObligationChecklistItem).filter(ObligationChecklistItem.user_id == user_id).all()
    return [_to_item_out(i) for i in items]


@router.post("/{user_id}/generate", response_model=list[ChecklistItemOut])
def generate_checklist(
    user_id: str,
    db: Session = Depends(get_db),
    gen_kwargs: dict = Depends(get_checklist_generator_kwargs),
) -> list[ChecklistItemOut]:
    """The manual activation endpoint — see module docstring. Runs the
    full profile -> traits -> category queries -> hybrid retrieval ->
    (optional) rerank -> LLM extraction -> deadline computation ->
    persisted-checklist pipeline synchronously, so the response IS the
    freshly generated checklist (not a job id to poll)."""
    try:
        items = checklist_service.generate_checklist_for_user(db, user_id, **gen_kwargs)
    except checklist_service.UserNotFoundError as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except NoProviderAvailableError as exc:
        # Caught here rather than left to bubble into a bare 500: this
        # is a routine, expected condition (no LLM key configured yet),
        # not a bug — the whole point of the manual-activation button
        # is that a person is watching it run live, so it should fail
        # with a message that tells them what to do next. Found by
        # actually running this endpoint against a live server with no
        # API key configured, not by reasoning about it in the abstract
        # — the test suite's fake-router fixtures never exercised this
        # path since they always supply a working fake LLM router.
        raise HTTPException(
            status_code=503,
            detail=(
                "Checklist generation needs at least one working LLM provider. "
                f"{exc} Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY "
                "in backend/.env and restart the backend."
            ),
        ) from exc
    return [_to_item_out(i) for i in items]


@router.patch("/{user_id}/items/{item_id}", response_model=ChecklistItemOut)
def update_checklist_item_status(
    user_id: str, item_id: str, payload: ChecklistItemStatusUpdate, db: Session = Depends(get_db)
) -> ChecklistItemOut:
    if payload.status not in _VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(_VALID_STATUSES)}")

    item = (
        db.query(ObligationChecklistItem)
        .filter(ObligationChecklistItem.id == item_id, ObligationChecklistItem.user_id == user_id)
        .first()
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Checklist item not found")

    item.status = payload.status
    db.commit()
    db.refresh(item)
    return _to_item_out(item)
