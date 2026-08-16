"""Unit tests for the hierarchical auto-chunker (chunker.py).

The most important property under test: a chunk NEVER splits a Khoản
mid-sentence — every sub-chunk boundary falls exactly at a Khoản start.
"""
from __future__ import annotations

import re

from app.ingestion.chunker import chunk_article, chunk_document
from app.ingestion.parser import parse_document

SHORT_TEXT = """\
Điều 1. Short article
1. A short clause.
2. Another short clause.
"""


def _khoan_starts(text: str) -> list[int]:
    return [m.start() for m in re.finditer(r"^\s*\d+\.\s", text, re.MULTILINE)]


def test_short_article_produces_a_single_chunk():
    doc = parse_document("doc", "Doc", SHORT_TEXT)
    article = doc.all_articles()[0]
    chunks = chunk_article(doc, article, max_chars=1500)
    assert len(chunks) == 1
    assert chunks[0].part_count == 1
    assert chunks[0].part_index == 0


def test_chunk_id_format_for_unsplit_article():
    doc = parse_document("sample-doc", "Doc", SHORT_TEXT)
    article = doc.all_articles()[0]
    chunks = chunk_article(doc, article, max_chars=1500)
    assert chunks[0].chunk_id == "sample-doc:dieu1"


def test_long_article_is_split_into_multiple_chunks():
    # Build an article whose body clearly exceeds a small max_chars.
    khoan_text = "Nội dung khoản dài được lặp lại nhiều lần để vượt ngưỡng độ dài tối đa cho phép. " * 3
    body_lines = [f"{i}. {khoan_text}" for i in range(1, 6)]
    text = "Điều 1. Long article\n" + "\n".join(body_lines) + "\n"

    doc = parse_document("doc", "Doc", text)
    article = doc.all_articles()[0]
    chunks = chunk_article(doc, article, max_chars=300)

    assert len(chunks) > 1
    assert all(c.part_count == len(chunks) for c in chunks)
    assert [c.part_index for c in chunks] == list(range(len(chunks)))


def test_no_chunk_splits_a_khoan_mid_sentence():
    """The core correctness property: every sub-chunk must begin at a
    genuine Khoản boundary — never partway through one clause's text."""
    khoan_text = "Nội dung khoản dài được lặp lại nhiều lần để vượt ngưỡng độ dài tối đa cho phép. " * 3
    body_lines = [f"{i}. {khoan_text}" for i in range(1, 8)]
    text = "Điều 1. Long article\n" + "\n".join(body_lines) + "\n"

    doc = parse_document("doc", "Doc", text)
    article = doc.all_articles()[0]
    chunks = chunk_article(doc, article, max_chars=350)

    for chunk in chunks:
        # Strip the heading line the chunker prepends, then confirm the
        # remaining text starts exactly at a "N. " Khoản marker.
        body_only = chunk.text.split("\n", 1)[1] if "\n" in chunk.text else ""
        assert re.match(r"^\d+\.\s", body_only.strip()), f"Chunk does not start at a Khoản boundary: {chunk.text[:80]!r}"


def test_heading_repeated_in_every_sub_chunk_for_context():
    khoan_text = "Nội dung khoản dài được lặp lại nhiều lần để vượt ngưỡng độ dài tối đa cho phép. " * 3
    body_lines = [f"{i}. {khoan_text}" for i in range(1, 6)]
    text = "Điều 7. My Article Title\n" + "\n".join(body_lines) + "\n"

    doc = parse_document("doc", "Doc", text)
    article = doc.all_articles()[0]
    chunks = chunk_article(doc, article, max_chars=300)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("Điều 7. My Article Title")


def test_chunk_document_covers_every_article():
    text = (
        "Chương I\nHEADING\n\n"
        "Điều 1. First\n1. Content one.\n\n"
        "Điều 2. Second\n1. Content two.\n\n"
        "Điều 3. Third\n1. Content three.\n"
    )
    doc = parse_document("doc", "Doc", text)
    chunks = chunk_document(doc, max_chars=1500)
    article_numbers = {c.article_number for c in chunks}
    assert article_numbers == {"1", "2", "3"}


def test_sub_chunk_ids_are_unique():
    khoan_text = "Nội dung khoản dài được lặp lại nhiều lần để vượt ngưỡng độ dài tối đa cho phép. " * 3
    body_lines = [f"{i}. {khoan_text}" for i in range(1, 6)]
    text = "Điều 1. Long article\n" + "\n".join(body_lines) + "\n"
    doc = parse_document("doc", "Doc", text)
    article = doc.all_articles()[0]
    chunks = chunk_article(doc, article, max_chars=300)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
