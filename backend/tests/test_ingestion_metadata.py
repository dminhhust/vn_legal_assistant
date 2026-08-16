"""Unit tests for metadata tagging (metadata.py)."""
from __future__ import annotations

import pytest

from app.ingestion.chunker import Chunk
from app.ingestion.metadata import SourceMeta, build_metadata, content_hash

_CHUNK = Chunk(
    chunk_id="doc:dieu1",
    document_id="doc",
    document_title="Test Doc",
    chapter_number="I",
    chapter_title="Chapter One",
    article_number="1",
    article_title="Article Title",
    text="Điều 1. Article Title\n1. Some content.",
    part_index=0,
    part_count=1,
)


def test_content_hash_is_stable_for_same_text():
    assert content_hash("hello") == content_hash("hello")


def test_content_hash_changes_when_text_changes():
    assert content_hash("hello") != content_hash("hello world")


def test_build_metadata_includes_all_expected_fields():
    source = SourceMeta(law_name="Test Law", category="tax", entity_type="individual")
    meta = build_metadata(_CHUNK, source)

    assert meta["document_id"] == "doc"
    assert meta["law_name"] == "Test Law"
    assert meta["category"] == "tax"
    assert meta["entity_type"] == "individual"
    assert meta["chapter_number"] == "I"
    assert meta["article_number"] == "1"
    assert meta["content_hash"] == content_hash(_CHUNK.text)


def test_unknown_category_raises():
    source = SourceMeta(law_name="Test Law", category="not_a_real_category")
    with pytest.raises(ValueError, match="Unknown category"):
        build_metadata(_CHUNK, source)


def test_unknown_entity_type_raises():
    source = SourceMeta(law_name="Test Law", category="tax", entity_type="alien")
    with pytest.raises(ValueError, match="Unknown entity_type"):
        build_metadata(_CHUNK, source)


def test_defaults_fill_in_sensibly():
    source = SourceMeta(law_name="Test Law", category="tax")  # entity_type defaults to "both"
    meta = build_metadata(_CHUNK, source)
    assert meta["entity_type"] == "both"
    assert meta["province_scope"] == "national"
    assert meta["effective_from"] == ""
    assert meta["effective_to"] == ""


def test_explicit_province_scope_overrides_national_default():
    source = SourceMeta(law_name="Test Law", category="tax", province_scope="Hanoi")
    meta = build_metadata(_CHUNK, source)
    assert meta["province_scope"] == "Hanoi"
