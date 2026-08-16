"""Adapter for the OpenAI API.

Docs: https://platform.openai.com/docs/overview
SDK:  pip install openai
"""
from __future__ import annotations

import json
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

# Override via OPENAI_MODEL in .env. Verify current valid model names
# against https://platform.openai.com/docs/models before relying on this
# default in prod.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError(self.name, "OPENAI_API_KEY not set", retryable=False)
        try:
            import openai
        except ImportError as exc:
            raise ProviderError(
                self.name, f"'openai' package not installed: {exc}", retryable=False
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)

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
        oa_messages = self._to_openai_messages(messages)
        oa_tools = [self._to_openai_tool(t) for t in (tools or [])] or None

        response_format = None
        if response_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        try:
            resp = self._client.chat.completions.create(
                model=model or DEFAULT_MODEL,
                messages=oa_messages,
                tools=oa_tools,
                response_format=response_format,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(self.name, str(exc)) from exc

        return self._from_openai_response(resp, response_schema is not None)

    # -- helpers -----------------------------------------------------

    @staticmethod
    def _to_openai_messages(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                out.append(
                    {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
                )
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    @staticmethod
    def _to_openai_tool(tool: ToolDefinition) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }

    def _from_openai_response(self, resp: Any, wants_structured: bool) -> LLMResponse:
        choice = resp.choices[0]
        msg = choice.message
        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                )
            )

        structured = None
        text = msg.content
        if wants_structured and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None

        usage = resp.usage
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            structured_output=structured,
            provider=self.name,
            model=resp.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=resp,
        )
