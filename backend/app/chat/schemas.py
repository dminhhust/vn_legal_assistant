"""Shared data structures and tool definitions for the RAG chatbot.

MVP SCOPE NOTE: the original prototype had a Supervisor Agent routing
between a Legal agent, a Personal Consultant agent, and a News/Product
agent, plus a three-layer memory architecture (short-term Redis buffer,
long-term episodic vector store, post-session reflection). This MVP
keeps exactly one agent — the RAG legal chatbot — since that's the
scope asked for; see docs/ARCHITECTURE.md "What this redesign cuts,
and why". Multi-turn context is handled the simple way: the frontend
resends prior turns as `history` on each request (see chat/router.py),
rather than a server-side session store — good enough for an MVP demo,
with a documented upgrade path if real session memory is wanted later.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.llm.schemas import ToolDefinition

SEARCH_LEGAL_TOOL = ToolDefinition(
    name="search_legal_obligations",
    description=(
        "Search the ingested Vietnamese legal corpus for information relevant to a "
        "free-text question about legal obligations, deadlines, or requirements."
    ),
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The search query."}},
        "required": ["query"],
    },
)

CHECKLIST_STATUS_TOOL = ToolDefinition(
    name="get_checklist_status",
    description=(
        "Get the user's personal obligation checklist items that are still pending, "
        "optionally filtered to only those due within a number of days."
    ),
    parameters={
        "type": "object",
        "properties": {
            "days_ahead": {
                "type": ["integer", "null"],
                "description": "Only include items due within this many days. Omit for all pending items.",
            }
        },
        "required": [],
    },
)

MARK_DONE_TOOL = ToolDefinition(
    name="mark_checklist_item_done",
    description="Mark one of the user's checklist items as done, matched by (partial) title text.",
    parameters={
        "type": "object",
        "properties": {
            "title_contains": {
                "type": "string",
                "description": "Text that appears in the item's title.",
            }
        },
        "required": ["title_contains"],
    },
)

ALL_CHAT_TOOLS = [SEARCH_LEGAL_TOOL, CHECKLIST_STATUS_TOOL, MARK_DONE_TOOL]


@dataclass
class LLMCallMeta:
    """Lightweight record of one LLM call the agent made — provider,
    model, and token counts, captured from the Model Router's
    LLMResponse. Kept for observability (log it, don't need a whole
    usage-tracking subsystem for an MVP)."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentResponse:
    text: str
    tool_calls_made: list[str] = field(default_factory=list)
    llm_calls: list[LLMCallMeta] = field(default_factory=list)
