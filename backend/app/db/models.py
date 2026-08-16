"""SQLAlchemy models: users, profiles, profile_history, trait_tags,
obligation_checklist_items.

MVP SCOPE NOTE: the original prototype also had NewsCacheItem,
ProductCacheItem, UsageEvent, and DailyBriefingCache tables, backing a
Personal Consultant + News/Product engine and a developer analytics
service. Neither is one of this MVP's four features (personal info
collection, auto-generated checklist with a manual activation
mechanism, RAG chatbot, legal-source crawler), so they're cut rather
than carried along unused — see docs/ARCHITECTURE.md.

Design notes (unchanged from the original design, still correct here):
  - `profiles` holds only the CURRENT snapshot (one row per user,
    overwritten on edit) — cheap to read for every request.
  - `profile_history` is append-only, one row per version, and is what
    makes trait changes auditable rather than just overwritten silently.
  - `trait_tags` is a normalized table (one row per tag) rather than a
    JSON array on `profiles`, because the legal RAG pipeline needs to
    query/filter by individual tags efficiently.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import JSON, Date, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    # Placeholder identity field until real auth (OAuth2) lands — MVP
    # only, no password.
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    profile: Mapped["Profile"] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    history: Mapped[list["ProfileHistory"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    traits: Mapped[list["TraitTag"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    checklist_items: Mapped[list["ObligationChecklistItem"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Profile(Base):
    """The CURRENT profile snapshot — one row per user."""

    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), unique=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)

    # --- Identity ---
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String, nullable=True)
    marital_status: Mapped[str | None] = mapped_column(String, nullable=True)
    province: Mapped[str | None] = mapped_column(String, nullable=True)
    dependents: Mapped[int] = mapped_column(Integer, default=0)

    # --- Work ---
    occupation_type: Mapped[str | None] = mapped_column(String, nullable=True)
    income_sources: Mapped[list] = mapped_column(JSON, default=list)
    has_business: Mapped[bool] = mapped_column(default=False)
    business_sector: Mapped[str | None] = mapped_column(String, nullable=True)

    # --- Assets ---
    owns_property: Mapped[bool] = mapped_column(default=False)
    owns_vehicle: Mapped[bool] = mapped_column(default=False)
    business_assets: Mapped[list] = mapped_column(JSON, default=list)

    # --- Preferences ---
    reminder_lead_days: Mapped[int] = mapped_column(Integer, default=3)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="profile")


class ProfileHistory(Base):
    """Append-only audit log — one row per version, never mutated."""

    __tablename__ = "profile_history"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    snapshot: Mapped[dict] = mapped_column(JSON)  # full profile field dict at this version
    changed_fields: Mapped[list] = mapped_column(JSON, default=list)  # vs. the previous version
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="history")


class TraitTag(Base):
    """Derived trait tags — recomputed wholesale (delete + reinsert) on
    every profile write. See app/profile/traits.py for the rules that
    produce these."""

    __tablename__ = "trait_tags"
    __table_args__ = (UniqueConstraint("user_id", "tag", name="uq_user_tag"),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)
    tag: Mapped[str] = mapped_column(String, index=True)
    source_rule: Mapped[str] = mapped_column(String)  # which rule function produced it
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="traits")


class ObligationChecklistItem(Base):
    """A single generated obligation checklist entry. One row per
    extracted obligation; the whole set is regenerated wholesale on
    each call to app.rag.checklist_service.generate_checklist_for_user
    (a known MVP-simple gap: this doesn't currently preserve user-set
    status like `done`/`dismissed` across a regeneration)."""

    __tablename__ = "obligation_checklist_items"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), index=True)

    title: Mapped[str] = mapped_column(String)
    category: Mapped[str] = mapped_column(String, index=True)
    description: Mapped[str] = mapped_column(String)

    # DeadlineRule fields, flattened (mirrors app/rag/schemas.py:DeadlineRule)
    deadline_type: Mapped[str] = mapped_column(String)
    deadline_month: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deadline_day: Mapped[int | None] = mapped_column(Integer, nullable=True)
    period_months: Mapped[int | None] = mapped_column(Integer, nullable=True)
    days_after_event: Mapped[int | None] = mapped_column(Integer, nullable=True)
    event_description: Mapped[str | None] = mapped_column(String, nullable=True)

    due_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    penalty_summary: Mapped[str] = mapped_column(String)
    source_citation: Mapped[str] = mapped_column(String)
    source_chunk_id: Mapped[str] = mapped_column(String)

    status: Mapped[str] = mapped_column(String, default="pending")  # pending/done/dismissed

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="checklist_items")
