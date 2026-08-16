"""Profile service: the business logic layer between the API routes
(router.py) and the DB models. Kept separate from the router so this
logic is callable from tests, a script, or an agent tool later, without
going through HTTP.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Profile, ProfileHistory, TraitTag, User
from app.profile.schemas import PROFILE_FIELD_NAMES, ProfileIn, ProfileOut, ProfileUpdate
from app.profile.traits import derive_traits


class UsernameTakenError(Exception):
    pass


class ProfileNotFoundError(Exception):
    pass


def _profile_to_dict(profile: Profile) -> dict:
    return {name: getattr(profile, name) for name in PROFILE_FIELD_NAMES}


def _recompute_traits(db: Session, user_id: str, profile_data: dict) -> list[str]:
    """Wholesale delete + reinsert — simplest correct approach for a
    prototype; a diffing update could replace this later if the traits
    table ever needs to preserve tag-level created_at timestamps across
    edits."""
    db.query(TraitTag).filter(TraitTag.user_id == user_id).delete()
    tag_pairs = derive_traits(profile_data)
    for tag, source_rule in tag_pairs:
        db.add(TraitTag(user_id=user_id, tag=tag, source_rule=source_rule))
    return [tag for tag, _ in tag_pairs]


def _to_profile_out(user: User, profile: Profile, traits: list[str]) -> ProfileOut:
    data = _profile_to_dict(profile)
    return ProfileOut(
        user_id=user.id,
        username=user.username,
        version=profile.version,
        traits=sorted(traits),
        **data,
    )


def create_profile(db: Session, payload: ProfileIn) -> ProfileOut:
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing is not None:
        raise UsernameTakenError(payload.username)

    user = User(username=payload.username)
    db.add(user)
    db.flush()  # populate user.id before creating dependent rows

    field_values = {name: getattr(payload, name) for name in PROFILE_FIELD_NAMES}
    profile = Profile(user_id=user.id, version=1, **field_values)
    db.add(profile)
    db.flush()

    db.add(
        ProfileHistory(
            user_id=user.id,
            version=1,
            snapshot=field_values,
            changed_fields=list(field_values.keys()),
        )
    )

    traits = _recompute_traits(db, user.id, field_values)
    db.commit()
    db.refresh(profile)

    return _to_profile_out(user, profile, traits)


def get_profile(db: Session, user_id: str) -> ProfileOut:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or user.profile is None:
        raise ProfileNotFoundError(user_id)
    traits = [t.tag for t in user.traits]
    return _to_profile_out(user, user.profile, traits)


def update_profile(db: Session, user_id: str, payload: ProfileUpdate) -> ProfileOut:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or user.profile is None:
        raise ProfileNotFoundError(user_id)

    profile = user.profile
    updates = payload.model_dump(exclude_unset=True)  # only fields explicitly sent

    changed_fields = []
    for field, value in updates.items():
        if getattr(profile, field) != value:
            setattr(profile, field, value)
            changed_fields.append(field)

    if changed_fields:
        profile.version += 1
        db.flush()
        field_values = _profile_to_dict(profile)
        db.add(
            ProfileHistory(
                user_id=user.id,
                version=profile.version,
                snapshot=field_values,
                changed_fields=changed_fields,
            )
        )
        traits = _recompute_traits(db, user.id, field_values)
    else:
        # No actual change (e.g. re-submitting the same value) — don't
        # bump the version or write a no-op history row.
        traits = [t.tag for t in user.traits]

    db.commit()
    db.refresh(profile)

    return _to_profile_out(user, profile, traits)
