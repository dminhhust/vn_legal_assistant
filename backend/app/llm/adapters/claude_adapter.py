"""Adapter for Anthropic's Claude API.

Docs: https://docs.claude.com/en/home
SDK:  pip install anthropic

Claude has no native "response_format=json_schema" the way OpenAI does.
The standard pattern (used here) is to define the requested schema as a
single tool and force the model to call it — the tool's `input` is then
the structured output.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from app.llm.adapters.base import ProviderAdapter
from app.llm.schemas import (
    LLMResponse,
    Message,
    ProviderError,
    ToolCall,
    ToolDefinition,
)

# Override via ANTHROPIC_MODEL in .env. Verify current valid model names
# against https://docs.claude.com before relying on this default in prod.
DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
_STRUCTURED_OUTPUT_TOOL_NAME = "emit_structured_output"


class ClaudeAdapter(ProviderAdapter):
    name = "claude"

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError(self.name, "ANTHROPIC_API_KEY not set", retryable=False)
        try:
            import anthropic
        except ImportError as exc:
            raise ProviderError(
                self.name, f"'anthropic' package not installed: {exc}", retryable=False
            ) from exc
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        response_schema: Optional[dict[str, Any]] = None,
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        system_text, claude_messages = self._to_claude_messages(messages)
        claude_tools = [self._to_claude_tool(t) for t in (tools or [])]

        forced_tool_choice = None
        if response_schema is not None:
            claude_tools.append(
                {
                    "name": _STRUCTURED_OUTPUT_TOOL_NAME,
                    "description": "Return the final answer in the required structured format.",
                    "input_schema": response_schema,
                }
            )
            forced_tool_choice = {"type": "tool", "name": _STRUCTURED_OUTPUT_TOOL_NAME}

        try:
            kwargs: dict[str, Any] = dict(
                model=model or DEFAULT_MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=claude_messages,
            )
            if system_text:
                kwargs["system"] = system_text
            if claude_tools:
                kwargs["tools"] = claude_tools
            if forced_tool_choice:
                kwargs["tool_choice"] = forced_tool_choice

            resp = self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize any SDK error
            raise ProviderError(self.name, str(exc)) from exc

        return self._from_claude_response(resp, response_schema is not None)

    # -- helpers -----------------------------------------------------

    @staticmethod
    def _to_claude_messages(messages: list[Message]) -> tuple[Optional[str], list[dict]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        out: list[dict] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": m.tool_call_id,
                                "content": m.content,
                            }
                        ],
                    }
                )
            else:
                out.append({"role": m.role, "content": m.content})
        return ("\n".join(system_parts) or None), out

    @staticmethod
    def _to_claude_tool(tool: ToolDefinition) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.parameters,
        }

    def _from_claude_response(self, resp: Any, wants_structured: bool) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        structured: Optional[dict] = None

        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                if wants_structured and block.name == _STRUCTURED_OUTPUT_TOOL_NAME:
                    structured = block.input
                else:
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input)
                    )

        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text="\n".join(text_parts) or None,
            tool_calls=tool_calls,
            structured_output=structured,
            provider=self.name,
            model=resp.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            raw=resp,
        )
