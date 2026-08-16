"""LLM-based structured obligation extraction (docs/ARCHITECTURE.md
§4.3). Runs PER CHUNK rather than batching a whole category's
candidates into one call — a few extra LLM calls is a cheap price for
every extracted obligation being unambiguously attributable to the one
Điều it actually came from, which matters a lot for a legal-adjacent
product where a wrong citation is worse than an extra API call.
"""
from __future__ import annotations

from typing import Optional

from app.llm.schemas import Message
from app.rag.retrieval import RetrievedChunk
from app.rag.schemas import OBLIGATION_EXTRACTION_SCHEMA, DeadlineRule, ObligationItem


def _build_citation(metadata: dict) -> str:
    law_name = metadata.get("law_name", "unknown source")
    article_number = metadata.get("article_number", "?")
    citation = f"{law_name}, Điều {article_number}"
    part_count = metadata.get("part_count", 1)
    if part_count and part_count > 1:
        part_index = metadata.get("part_index", 0)
        citation += f" (part {part_index + 1}/{part_count})"
    return citation


def extract_obligations_from_chunk(
    chunk: RetrievedChunk,
    category: str,
    *,
    router=None,
    task: str = "legal_extraction",
) -> list[ObligationItem]:
    """Extracts every obligation explicitly stated in ONE chunk. Returns
    an empty list if the chunk states no obligation — extraction should
    never invent one where none exists."""
    if router is None:
        from app.llm.router import get_router

        router = get_router()

    prompt = (
        "Below is one excerpt (a single legal article) from a Vietnamese "
        f"legal document, tagged under the category '{category}'. Extract "
        "every distinct legal obligation explicitly stated in this excerpt "
        "as a JSON object per the required schema. Be conservative: only "
        "extract what is explicitly stated in the text below — never invent "
        "an obligation, deadline, or penalty that isn't present. If the "
        "excerpt states no obligation, return an empty list.\n\n"
        f"{chunk.text}"
    )

    response = router.complete(
        [Message(role="user", content=prompt)],
        task=task,
        response_schema=OBLIGATION_EXTRACTION_SCHEMA,
        max_tokens=1024,
    )
    raw_obligations = (response.structured_output or {}).get("obligations", [])

    citation = _build_citation(chunk.metadata)
    items = []
    for raw in raw_obligations:
        deadline_rule = DeadlineRule(
            type=raw["deadline_type"],
            month=raw.get("deadline_month"),
            day=raw.get("deadline_day"),
            period_months=raw.get("period_months"),
            days_after_event=raw.get("days_after_event"),
            event_description=raw.get("event_description"),
        )
        items.append(
            ObligationItem(
                title=raw["title"],
                category=category,
                description=raw["description"],
                deadline_rule=deadline_rule,
                penalty_summary=raw["penalty_summary"],
                source_citation=citation,
                source_chunk_id=chunk.chunk_id,
            )
        )
    return items


def extract_obligations_for_category(
    chunks: list[RetrievedChunk],
    category: str,
    *,
    router=None,
    top_k_chunks: int = 3,
) -> list[ObligationItem]:
    """Runs per-chunk extraction over the top `top_k_chunks` most
    relevant retrieved chunks for one category — bounding LLM calls per
    category rather than extracting from every candidate the retriever
    found."""
    all_items: list[ObligationItem] = []
    for chunk in chunks[:top_k_chunks]:
        all_items.extend(extract_obligations_from_chunk(chunk, category, router=router))
    return all_items
