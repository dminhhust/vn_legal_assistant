"""Unit tests for the structure-aware parser (Chương/Điều detection)."""
from __future__ import annotations

from app.ingestion.parser import parse_document

SIMPLE_TEXT = """\
Chương I
QUY ĐỊNH CHUNG

Điều 1. Mục đích
1. Nội dung khoản một.
2. Nội dung khoản hai.

Điều 2. Phạm vi áp dụng
Nội dung không đánh số khoản.

Chương II
NGHĨA VỤ

Điều 3. Nghĩa vụ báo cáo
1. Nội dung khoản một của điều ba.
"""


def test_parses_two_chapters():
    doc = parse_document("test-doc", "Test Document", SIMPLE_TEXT)
    assert len(doc.chapters) == 2
    assert doc.chapters[0].number == "I"
    assert doc.chapters[1].number == "II"


def test_articles_assigned_to_correct_chapter():
    doc = parse_document("test-doc", "Test Document", SIMPLE_TEXT)
    chapter_1_articles = [a.number for a in doc.chapters[0].articles]
    chapter_2_articles = [a.number for a in doc.chapters[1].articles]
    assert chapter_1_articles == ["1", "2"]
    assert chapter_2_articles == ["3"]


def test_article_title_captured():
    doc = parse_document("test-doc", "Test Document", SIMPLE_TEXT)
    article_1 = doc.chapters[0].articles[0]
    assert article_1.title == "Mục đích"


def test_article_body_contains_khoan_text():
    doc = parse_document("test-doc", "Test Document", SIMPLE_TEXT)
    article_1 = doc.chapters[0].articles[0]
    assert "Nội dung khoản một" in article_1.body
    assert "Nội dung khoản hai" in article_1.body


def test_article_without_numbered_khoan_still_captures_body():
    doc = parse_document("test-doc", "Test Document", SIMPLE_TEXT)
    article_2 = doc.chapters[0].articles[1]
    assert "Nội dung không đánh số khoản" in article_2.body


def test_all_articles_returns_articles_across_chapters_in_order():
    doc = parse_document("test-doc", "Test Document", SIMPLE_TEXT)
    numbers = [a.number for a in doc.all_articles()]
    assert numbers == ["1", "2", "3"]


def test_articles_before_any_chapter_go_to_preamble():
    text = "Điều 1. Preamble article\n1. Some content.\n\nChương I\nHEADING\n\nĐiều 2. Chaptered article\n"
    doc = parse_document("test-doc", "Test Document", text)
    assert len(doc.preamble_articles) == 1
    assert doc.preamble_articles[0].number == "1"
    assert len(doc.chapters) == 1
    assert doc.chapters[0].articles[0].number == "2"


def test_document_with_no_chapters_at_all():
    text = "Điều 1. Only article\n1. Content.\n"
    doc = parse_document("test-doc", "Test Document", text)
    assert doc.chapters == []
    assert len(doc.preamble_articles) == 1


def test_empty_text_produces_empty_document():
    doc = parse_document("test-doc", "Test Document", "")
    assert doc.chapters == []
    assert doc.preamble_articles == []
    assert doc.all_articles() == []


def test_arabic_numeral_chapter_supported():
    text = "Chương 1\nHEADING\n\nĐiều 1. Article\n1. Content.\n"
    doc = parse_document("test-doc", "Test Document", text)
    assert doc.chapters[0].number == "1"
