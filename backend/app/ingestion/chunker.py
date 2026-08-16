"""Hierarchical auto-chunker.

Default rule: ONE CHUNK PER ĐIỀU (article). This preserves legal
meaning — each Điều is drafted as a self-contained obligation unit —
and naturally caps chunk size, since individual Điều are rarely huge.

Fallback: if an Điều's full text exceeds `max_chars`, a secondary
size-based sub-chunker splits it along Khoản ("1.", "2.", ...)
boundaries instead of an arbitrary character cut, so a chunk never
splits a Khoản mid-sentence. Each sub-chunk repeats the Điều's heading
as a context prefix (the "small-to-big" RAG pattern from
docs/ARCHITECTURE.md §4.5) so a retrieved sub-chunk is still
self-orienting even without its siblings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from app.ingestion.models import ParsedArticle, ParsedDocument

# Conservative default — real embedding models handle far more, but this
# keeps each chunk tightly scoped to roughly one legal idea rather than
# an entire long article's worth of unrelated clauses.
DEFAULT_MAX_CHARS = 1500

# Matches the START of a Khoản line ("1. ...", "12. ...") so splitting
# never cuts into the middle of a clause's text.
_KHOAN_SPLIT_RE = re.compile(r"(?=^\s*\d+\.\s)", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_title: str
    chapter_number: Optional[str]
    chapter_title: Optional[str]
    article_number: str
    article_title: str
    text: str  # the actual text to embed — includes a heading prefix for context
    part_index: int  # 0 for a whole-article chunk; 0..N-1 for sub-chunks
    part_count: int  # 1 if not split; total sub-chunk count otherwise


def _full_article_text(article: ParsedArticle) -> str:
    heading = f"Điều {article.number}. {article.title}".strip()
    return f"{heading}\n{article.body}".strip()


def chunk_article(
    document: ParsedDocument, article: ParsedArticle, max_chars: int = DEFAULT_MAX_CHARS
) -> list[Chunk]:
    full_text = _full_article_text(article)

    if len(full_text) <= max_chars:
        return [
            Chunk(
                chunk_id=f"{document.doc_id}:dieu{article.number}",
                document_id=document.doc_id,
                document_title=document.title,
                chapter_number=article.chapter_number,
                chapter_title=article.chapter_title,
                article_number=article.number,
                article_title=article.title,
                text=full_text,
                part_index=0,
                part_count=1,
            )
        ]

    # Fallback: split along Khoản boundaries and greedily group
    # consecutive Khoản into sub-chunks under max_chars, never breaking
    # a single Khoản across two sub-chunks.
    heading = f"Điều {article.number}. {article.title}".strip()
    khoan_parts = [p for p in _KHOAN_SPLIT_RE.split(article.body) if p.strip()]
    if not khoan_parts:
        # No detectable Khoản structure to split on (unusual, but
        # possible for a short-form article) — keep one oversized chunk
        # rather than silently cutting mid-sentence at an arbitrary point.
        khoan_parts = [article.body]

    grouped: list[str] = []
    current = ""
    for part in khoan_parts:
        candidate = f"{current}\n{part}".strip() if current else part.strip()
        if current and len(heading) + len(candidate) > max_chars:
            grouped.append(current)
            current = part.strip()
        else:
            current = candidate
    if current:
        grouped.append(current)

    return [
        Chunk(
            chunk_id=f"{document.doc_id}:dieu{article.number}:part{i}",
            document_id=document.doc_id,
            document_title=document.title,
            chapter_number=article.chapter_number,
            chapter_title=article.chapter_title,
            article_number=article.number,
            article_title=article.title,
            text=f"{heading}\n{group_text}".strip(),
            part_index=i,
            part_count=len(grouped),
        )
        for i, group_text in enumerate(grouped)
    ]


def chunk_document(document: ParsedDocument, max_chars: int = DEFAULT_MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    for article in document.all_articles():
        chunks.extend(chunk_article(document, article, max_chars=max_chars))
    return chunks
