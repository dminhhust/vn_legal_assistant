"""Pydantic request/response schemas for the Profile Service.

MVP SCOPE NOTE: the original prototype collected a much wider profile
(health/lifestyle, interests, notification preferences) to feed a
Personal Consultant + News/Product engine that this MVP redesign does
not include (see docs/ARCHITECTURE.md "What this redesign cuts, and
why"). Every field kept here is a field that actually drives something
in this MVP — legal-category applicability (app/rag/query_builder.py)
via derived trait tags (app/profile/traits.py). Collecting fields
nothing reads is a trust problem for a legal-adjacent product, not just
wasted onboarding time, so they were removed rather than kept "for
later."

Enums are plain `Literal`s rather than DB-level enum types, so adding a
new option is a one-line change here, not a migration.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

OccupationType = Literal["employee", "freelancer", "business_owner", "student", "retired", "unemployed"]

# Single source of truth for "which fields make up a profile" — reused
# by the service layer to move data between the ORM model and the API
# schema without hand-listing field names in multiple places.
PROFILE_FIELD_NAMES: list[str] = [
    "age", "gender", "marital_status", "province", "dependents",
    "occupation_type", "income_sources", "has_business", "business_sector",
    "owns_property", "owns_vehicle", "business_assets",
    "reminder_lead_days",
]


class ProfileIn(BaseModel):
    """Full onboarding payload. Only `username` is required — every
    other field defaults sensibly so onboarding doesn't feel like a
    wall of mandatory questions; the Streamlit wizard groups these into
    steps, but the API itself accepts them all at once."""

    username: str = Field(..., min_length=1, max_length=64)

    # Identity — drives residence/family-obligation applicability
    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    province: Optional[str] = None
    dependents: int = 0

    # Work — drives tax + labor + business-obligation applicability
    occupation_type: Optional[OccupationType] = None
    income_sources: list[str] = Field(default_factory=list)
    has_business: bool = False
    business_sector: Optional[str] = None

    # Assets — drives property/vehicle-obligation applicability
    owns_property: bool = False
    owns_vehicle: bool = False
    business_assets: list[str] = Field(default_factory=list)

    # Preferences
    reminder_lead_days: int = 3


class ProfileUpdate(BaseModel):
    """Partial update — every field optional. The service layer uses
    `model_dump(exclude_unset=True)` so a field omitted from the request
    is left untouched, never accidentally reset to null."""

    age: Optional[int] = Field(None, ge=0, le=120)
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    province: Optional[str] = None
    dependents: Optional[int] = None

    occupation_type: Optional[OccupationType] = None
    income_sources: Optional[list[str]] = None
    has_business: Optional[bool] = None
    business_sector: Optional[str] = None

    owns_property: Optional[bool] = None
    owns_vehicle: Optional[bool] = None
    business_assets: Optional[list[str]] = None

    reminder_lead_days: Optional[int] = None


class ProfileOut(BaseModel):
    user_id: str
    username: str
    version: int
    traits: list[str]

    age: Optional[int] = None
    gender: Optional[str] = None
    marital_status: Optional[str] = None
    province: Optional[str] = None
    dependents: int

    occupation_type: Optional[str] = None
    income_sources: list[str]
    has_business: bool
    business_sector: Optional[str] = None

    owns_property: bool
    owns_vehicle: bool
    business_assets: list[str]

    reminder_lead_days: int
