"""Structure-aware parser for Vietnamese legal documents.

Recognizes the standard hierarchy used in Vietnamese law:

    Luật / Nghị định / Thông tư  (the document itself)
      > Chương        (chapter — roman or arabic numeral)
        > Điều         (article — the natural chunk boundary; see chunker.py)
          > Khoản       (clause, "1.", "2.", ...)
            > Điểm       (point, "a)", "b)", ...)

The parser only needs to reliably find CHAPTER and ARTICLE boundaries.
Finer Khoản/Điểm structure is preserved verbatim as raw text inside
each article's `body` and is only walked by the chunker's size-based
fallback (chunker.py) when a single article is unusually long.
"""
from __future__ import annotations

import re

from app.ingestion.models import ParsedArticle, ParsedChapter, ParsedDocument

# "Chương I", "Chương 2", "Chương IV." — roman or arabic numeral, optional trailing period.
_CHAPTER_RE = re.compile(r"^\s*Ch(?:ươ|uơ)ng\s+([IVXLCDM]+|\d+)\.?\s*(.*)$", re.IGNORECASE)
# "Điều 1.", "Điều 12:" — always an arabic numeral in real Vietnamese legal drafting.
_ARTICLE_RE = re.compile(r"^\s*Điều\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)


def parse_document(doc_id: str, title: str, raw_text: str) -> ParsedDocument:
    """Parses raw legal text into a ParsedDocument tree.

    Lines before the first Điều that aren't a Chương heading (e.g. a
    "Căn cứ ..." legal-basis preamble) are intentionally dropped from
    the structured output — they carry no independently queryable
    obligation content, and keeping them would create a chunk with no
    real article identity.
    """
    doc = ParsedDocument(doc_id=doc_id, title=title)

    current_chapter: ParsedChapter | None = None
    current_article: dict | None = None  # {"number", "title", "lines": [...]}

    def flush_article() -> None:
        nonlocal current_article
        if current_article is None:
            return
        article = ParsedArticle(
            number=current_article["number"],
            title=current_article["title"],
            body="\n".join(current_article["lines"]).strip(),
            chapter_number=current_chapter.number if current_chapter else None,
            chapter_title=current_chapter.title if current_chapter else None,
        )
        if current_chapter is not None:
            current_chapter.articles.append(article)
        else:
            doc.preamble_articles.append(article)
        current_article = None

    for line in raw_text.splitlines():
        chapter_match = _CHAPTER_RE.match(line)
        article_match = _ARTICLE_RE.match(line)

        if chapter_match:
            flush_article()
            current_chapter = ParsedChapter(
                number=chapter_match.group(1), title=chapter_match.group(2).strip()
            )
            doc.chapters.append(current_chapter)
            continue

        if article_match:
            flush_article()
            current_article = {
                "number": article_match.group(1),
                "title": article_match.group(2).strip(),
                "lines": [],
            }
            continue

        if current_article is not None:
            current_article["lines"].append(line)

    flush_article()
    return doc
