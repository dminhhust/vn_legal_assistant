"""FastAPI routes for the Profile Service (Phase 1).

See docs/ARCHITECTURE.md §4.1 and docs/IMPLEMENTATION_PLAN.md Phase 1.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.profile import service
from app.profile.schemas import ProfileIn, ProfileOut, ProfileUpdate

router = APIRouter(prefix="/profile", tags=["profile"])


@router.post("", response_model=ProfileOut, status_code=201)
def onboard(payload: ProfileIn, db: Session = Depends(get_db)) -> ProfileOut:
    try:
        return service.create_profile(db, payload)
    except service.UsernameTakenError as exc:
        raise HTTPException(
            status_code=409, detail=f"Username '{exc}' is already taken"
        ) from exc


@router.get("/{user_id}", response_model=ProfileOut)
def read_profile(user_id: str, db: Session = Depends(get_db)) -> ProfileOut:
    try:
        return service.get_profile(db, user_id)
    except service.ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Profile not found") from exc


@router.put("/{user_id}", response_model=ProfileOut)
def edit_profile(
    user_id: str, payload: ProfileUpdate, db: Session = Depends(get_db)
) -> ProfileOut:
    try:
        return service.update_profile(db, user_id, payload)
    except service.ProfileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Profile not found") from exc
