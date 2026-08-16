"""Data structures produced by the parser (parser.py) and consumed by
the chunker (chunker.py). Kept deliberately minimal — the parser only
needs to reliably capture Chương/Điều boundaries; finer Khoản/Điểm
structure stays as raw text inside each article's body and is only
walked by the chunker's size-based fallback for unusually long
articles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedArticle:
    number: str  # e.g. "5" (as it appears after "Điều")
    title: str  # heading text on the same line as "Điều N."
    body: str  # full raw text of everything until the next Điều/Chương
    chapter_number: Optional[str] = None
    chapter_title: Optional[str] = None


@dataclass
class ParsedChapter:
    number: str  # e.g. "I" or "1"
    title: str
    articles: list[ParsedArticle] = field(default_factory=list)


@dataclass
class ParsedDocument:
    doc_id: str  # stable identifier for the source law, e.g. "sample-test-law"
    title: str
    chapters: list[ParsedChapter] = field(default_factory=list)
    # Articles appearing before any Chương heading (some laws have no
    # chapters at all, or only a short preamble) live here instead.
    preamble_articles: list[ParsedArticle] = field(default_factory=list)

    def all_articles(self) -> list[ParsedArticle]:
        arts = list(self.preamble_articles)
        for ch in self.chapters:
            arts.extend(ch.articles)
        return arts
