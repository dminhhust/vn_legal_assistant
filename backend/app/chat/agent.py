"""The RAG legal chatbot agent. Wraps the legal RAG retrieval stack and
the checklist DB as callable tools, and runs a genuine tool-use loop:
ask the LLM which tool(s) it needs given the user's message, execute
them for real, then ask for a final, citation-grounded answer.

SIMPLIFICATION — documented rather than hidden (carried over from the
original codebase's own honest note, still true here): this feeds tool
results back as a fresh user-role follow-up message rather than
reconstructing a provider-native multi-turn tool-call history.
app.llm.schemas.Message doesn't carry an assistant-side `tool_calls`
field, so a fully protocol-native multi-turn tool conversation isn't
representable yet. The pattern below still exercises genuine
provider-native tool-selection (the first call, where the model decides
WHICH tool(s) to invoke) and produces an answer genuinely grounded in
real tool output — it's the "one assistant turn per network round-trip"
shape that's simplified, not the tool-selection or grounding itself. A
real fix (adding `tool_calls` to `Message` and updating the three
adapters to round-trip it) is a good next hardening step, not done here
to keep this MVP's diff focused on the four requested features.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy.orm import Session

from app.chat.schemas import ALL_CHAT_TOOLS, AgentResponse, LLMCallMeta
from app.chat.tools import get_checklist_status, mark_checklist_item_done, search_legal_obligations
from app.ingestion.embeddings import EmbeddingProvider
from app.ingestion.vector_store import VectorStoreWriter
from app.llm.schemas import Message, ToolCall

SYSTEM_PROMPT = (
    "You are a legal-obligation assistant for a personal legal assistant app focused "
    "on Vietnamese law. You have tools to search a legal knowledge base and to check "
    "or update the user's personal obligation checklist. Always cite sources when you "
    "state a legal fact, using the citations the search tool provides. Never state a "
    "legal fact you cannot support from a tool result — say you don't have enough "
    "information rather than guessing. When you answer a substantive legal question, "
    "remind the user this is not a substitute for professional legal advice."
)


def _to_call_meta(response) -> LLMCallMeta:
    return LLMCallMeta(
        provider=response.provider,
        model=response.model,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


class LegalChatAgent:
    def __init__(
        self,
        router=None,
        vector_store: Optional[VectorStoreWriter] = None,
        embedder: Optional[EmbeddingProvider] = None,
    ):
        self._router = router  # injectable for tests; falls back to the real Model Router
        self._vector_store = vector_store
        self._embedder = embedder

    def handle(
        self,
        message: str,
        *,
        user_id: str,
        db: Session,
        history: Optional[list[tuple[str, str]]] = None,
    ) -> AgentResponse:
        """`history` is an optional list of (role, content) pairs from
        earlier in the conversation — the client resends it each turn
        (see chat/router.py). This keeps the MVP stateless server-side;
        there's no session store to run or reason about."""
        router = self._get_router()

        history_messages = [Message(role=role, content=content) for role, content in (history or [])]

        first = router.complete(
            [
                Message(role="system", content=SYSTEM_PROMPT),
                *history_messages,
                Message(role="user", content=message),
            ],
            tools=ALL_CHAT_TOOLS,
            task="chat",
        )
        llm_calls = [_to_call_meta(first)]

        if not first.tool_calls:
            # No tool needed (e.g. a clarifying question, or a general
            # greeting). The system prompt already constrains the model
            # not to state ungrounded legal facts even without tool use.
            return AgentResponse(text=first.text or "", tool_calls_made=[], llm_calls=llm_calls)

        tool_calls_made: list[str] = []
        result_blocks: list[str] = []
        for call in first.tool_calls:
            result_text = self._execute_tool(call, user_id=user_id, db=db)
            tool_calls_made.append(call.name)
            result_blocks.append(f"[{call.name} result]\n{result_text}")

        follow_up_prompt = (
            f"Original user question: {message}\n\n"
            "Tool results:\n" + "\n\n".join(result_blocks) + "\n\n"
            "Using ONLY the tool results above, write a clear, direct answer to "
            "the user's original question. Cite sources where the tool results "
            "provide them. If the tool results don't fully answer the question, "
            "say what's missing rather than guessing."
        )
        second = router.complete(
            [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=follow_up_prompt)],
            task="chat",
        )
        llm_calls.append(_to_call_meta(second))
        final_text = second.text or ""
        return AgentResponse(text=final_text, tool_calls_made=tool_calls_made, llm_calls=llm_calls)

    def _get_router(self):
        if self._router is not None:
            return self._router
        from app.llm.router import get_router

        return get_router()

    def _execute_tool(self, call: ToolCall, *, user_id: str, db: Session) -> str:
        if call.name == "search_legal_obligations":
            return search_legal_obligations(
                call.arguments.get("query", ""),
                user_id=user_id,
                db=db,
                vector_store=self._vector_store,
                embedder=self._embedder,
            )
        if call.name == "get_checklist_status":
            return get_checklist_status(call.arguments.get("days_ahead"), user_id=user_id, db=db)
        if call.name == "mark_checklist_item_done":
            return mark_checklist_item_done(
                call.arguments.get("title_contains", ""), user_id=user_id, db=db
            )
        return f"Unknown tool requested: {call.name}"
