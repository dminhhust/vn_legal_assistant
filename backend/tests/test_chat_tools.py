"""Unit tests for the RAG chatbot's tool implementations (tools.py) —
no LLM involved, direct calls against a real (SQLite) DB and a real
(in-memory Chroma) vector store."""
from __future__ import annotations

import uuid
from datetime import date, timedelta

import chromadb
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.chat.tools import get_checklist_status, mark_checklist_item_done, search_legal_obligations
from app.db.models import ObligationChecklistItem
from app.db.session import Base
from app.ingestion.embeddings import HashingEmbeddingProvider
from app.ingestion.metadata import SourceMeta
from app.ingestion.pipeline import ingest_document
from app.ingestion.vector_store import VectorStoreWriter
from app.profile.schemas import ProfileIn
from app.profile.service import create_profile


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
def embedder():
    return HashingEmbeddingProvider()


@pytest.fixture()
def vector_store():
    return VectorStoreWriter(client=chromadb.EphemeralClient(), collection_name=f"test-{uuid.uuid4().hex}")


@pytest.fixture()
def user_profile(db_session):
    return create_profile(
        db_session, ProfileIn(username="tooluser", has_business=True, business_sector="retail")
    )


def _add_checklist_item(db_session, user_id, title, category, due_date, status="pending"):
    item = ObligationChecklistItem(
        user_id=user_id,
        title=title,
        category=category,
        description="desc",
        deadline_type="fixed",
        due_date=due_date,
        penalty_summary="penalty",
        source_citation="Test Law, Điều 1",
        source_chunk_id="test:dieu1",
        status=status,
    )
    db_session.add(item)
    db_session.commit()
    db_session.refresh(item)
    return item


class TestGetChecklistStatus:
    def test_returns_all_pending_items_when_no_days_ahead(self, db_session, user_profile):
        _add_checklist_item(db_session, user_profile.user_id, "File taxes", "tax", date(2026, 3, 31))
        _add_checklist_item(
            db_session, user_profile.user_id, "Renew license", "business_licensing", date(2026, 6, 1)
        )

        result = get_checklist_status(None, user_id=user_profile.user_id, db=db_session)

        assert "File taxes" in result
        assert "Renew license" in result

    def test_filters_by_days_ahead(self, db_session, user_profile):
        _add_checklist_item(
            db_session, user_profile.user_id, "Due soon", "tax", date.today() + timedelta(days=5)
        )
        _add_checklist_item(
            db_session, user_profile.user_id, "Due later", "tax", date.today() + timedelta(days=60)
        )

        result = get_checklist_status(10, user_id=user_profile.user_id, db=db_session)

        assert "Due soon" in result
        assert "Due later" not in result

    def test_excludes_done_items(self, db_session, user_profile):
        _add_checklist_item(
            db_session, user_profile.user_id, "Already done", "tax", date(2026, 1, 1), status="done"
        )

        result = get_checklist_status(None, user_id=user_profile.user_id, db=db_session)

        assert "Already done" not in result

    def test_no_items_returns_friendly_message(self, db_session, user_profile):
        result = get_checklist_status(None, user_id=user_profile.user_id, db=db_session)
        assert "No pending checklist items" in result


class TestMarkChecklistItemDone:
    def test_marks_matching_item_done(self, db_session, user_profile):
        item = _add_checklist_item(
            db_session, user_profile.user_id, "File annual tax return", "tax", date(2026, 3, 31)
        )

        result = mark_checklist_item_done("annual tax", user_id=user_profile.user_id, db=db_session)

        db_session.refresh(item)
        assert item.status == "done"
        assert "File annual tax return" in result

    def test_no_match_returns_friendly_message(self, db_session, user_profile):
        result = mark_checklist_item_done(
            "nonexistent item", user_id=user_profile.user_id, db=db_session
        )
        assert "No checklist item found" in result

    def test_ambiguous_match_asks_for_clarification(self, db_session, user_profile):
        _add_checklist_item(db_session, user_profile.user_id, "Tax filing A", "tax", date(2026, 3, 31))
        _add_checklist_item(db_session, user_profile.user_id, "Tax filing B", "tax", date(2026, 4, 30))

        result = mark_checklist_item_done("Tax filing", user_id=user_profile.user_id, db=db_session)

        assert "Multiple items match" in result

    def test_marking_done_does_not_affect_other_users_items(self, db_session):
        user_a = create_profile(db_session, ProfileIn(username="user_a"))
        user_b = create_profile(db_session, ProfileIn(username="user_b"))
        item_b = _add_checklist_item(db_session, user_b.user_id, "Shared title text", "tax", date(2026, 1, 1))

        result = mark_checklist_item_done("Shared title", user_id=user_a.user_id, db=db_session)

        db_session.refresh(item_b)
        assert "No checklist item found" in result
        assert item_b.status == "pending"


class TestSearchLegalObligations:
    def test_returns_relevant_chunk_with_citation(self, db_session, user_profile, vector_store, embedder):
        ingest_document(
            "tax-doc",
            "Tax Doc",
            "Điều 1. Personal income tax filing\n1. Individuals must file an annual tax return by March 31.\n",
            SourceMeta(law_name="Test Tax Law", category="tax", entity_type="both"),
            vector_store=vector_store,
            embedder=embedder,
        )

        result = search_legal_obligations(
            "personal income tax annual filing",
            user_id=user_profile.user_id,
            db=db_session,
            vector_store=vector_store,
            embedder=embedder,
        )

        assert "Test Tax Law" in result
        assert "Điều 1" in result

    def test_no_data_returns_friendly_message(self, db_session, user_profile, vector_store, embedder):
        result = search_legal_obligations(
            "anything at all",
            user_id=user_profile.user_id,
            db=db_session,
            vector_store=vector_store,
            embedder=embedder,
        )
        assert "No relevant legal information found" in result

    def test_unknown_user_returns_friendly_message(self, db_session, vector_store, embedder):
        result = search_legal_obligations(
            "anything", user_id="does-not-exist", db=db_session, vector_store=vector_store, embedder=embedder
        )
        assert "No profile found" in result
