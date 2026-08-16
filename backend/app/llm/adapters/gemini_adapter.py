"""Adapter for Google AI Studio's Gemini API.

Docs: https://ai.google.dev/gemini-api/docs
SDK:  pip install google-genai

Note: Google's Python SDK naming/surface has shifted before (the older
`google-generativeai` package vs. the newer unified `google-genai`
client used below). If this adapter errors on import or on the
`generate_content` call shape, check the current SDK docs — because this
logic is isolated to one file, fixing it never touches the Router,
other adapters, or any agent code.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
from typing import Any, Optional, Union

from app.llm.adapters.base import ProviderAdapter
from app.llm.schemas import (
    LLMResponse,
    Message,
    ProviderError,
    ToolCall,
    ToolDefinition,
)

# Override via GOOGLE_MODEL in .env. Verify current valid model names
# against https://ai.google.dev/gemini-api/docs/models before relying on
# this default in prod.
DEFAULT_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")

# Multiple keys can be supplied as a comma-separated list in GOOGLE_API_KEYS
# (preferred) or GOOGLE_API_KEY (kept for backwards compatibility with the
# single-key setup). Whitespace around each key is stripped; empty entries
# are dropped.
_KEYS_ENV_VARS = ("GOOGLE_API_KEYS", "GOOGLE_API_KEY")


def _parse_keys(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def _keys_from_env() -> list[str]:
    for var in _KEYS_ENV_VARS:
        raw = os.getenv(var)
        if raw:
            keys = _parse_keys(raw)
            if keys:
                return keys
    return []


class GeminiAdapter(ProviderAdapter):
    name = "gemini"

    def __init__(self, api_key: Optional[Union[str, list[str]]] = None):
        if api_key is None:
            keys = _keys_from_env()
        elif isinstance(api_key, str):
            keys = _parse_keys(api_key) if "," in api_key else [api_key]
        else:
            keys = [k for k in api_key if k]

        if not keys:
            raise ProviderError(
                self.name,
                "no API key(s) set (checked GOOGLE_API_KEYS / GOOGLE_API_KEY)",
                retryable=False,
            )

        try:
            from google import genai
        except ImportError as exc:
            raise ProviderError(
                self.name, f"'google-genai' package not installed: {exc}", retryable=False
            ) from exc

        self._keys = keys
        # One client per key, built once up front so rotation is just an
        # index swap with no re-auth cost on the hot path.
        self._clients = [genai.Client(api_key=k) for k in keys]
        self._lock = threading.Lock()
        self._cycle = itertools.cycle(range(len(self._clients)))
        # Start each adapter instance on a random-ish offset so parallel
        # adapter instances don't all hammer key[0] first; simplest correct
        # approach is just to advance the cycle once per instance.

    def _next_client_index(self) -> int:
        with self._lock:
            return next(self._cycle)

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
        system_text, contents = self._to_gemini_contents(messages)
        resolved_model = model or DEFAULT_MODEL

        config: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_text:
            config["system_instruction"] = system_text
        if tools:
            config["tools"] = [self._to_gemini_tool(t) for t in tools]
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = self._sanitize_schema_for_gemini(response_schema)

        # Try each key at most once per call, starting from the next key in
        # rotation order. On a retryable failure (e.g. rate limit / quota
        # exhausted on that particular key) we rotate to the next one and
        # retry the same request; on a non-retryable failure we raise
        # immediately rather than burning through the rest of the keys.
        last_exc: Optional[Exception] = None
        for _ in range(len(self._clients)):
            idx = self._next_client_index()
            client = self._clients[idx]
            try:
                resp = client.models.generate_content(
                    model=resolved_model, contents=contents, config=config
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                provider_exc = ProviderError(self.name, str(exc))
                if not getattr(provider_exc, "retryable", True):
                    raise provider_exc from exc
                continue
            else:
                return self._from_gemini_response(resp, response_schema is not None, resolved_model)

        raise ProviderError(
            self.name,
            f"all {len(self._clients)} key(s) exhausted; last error: {last_exc}",
        ) from last_exc

    # -- helpers -----------------------------------------------------

    # Gemini's Schema model is stricter than generic JSON Schema: `type`
    # must be a SINGLE uppercase enum value (STRING/INTEGER/OBJECT/...)
    # and nullable fields are expressed via `nullable: true`, NOT via
    # multi-type arrays like ["integer", "null"] — which this codebase's
    # shared schemas (e.g. app.rag.schemas.OBLIGATION_EXTRACTION_SCHEMA)
    # legitimately use for OpenAI/Claude. Sending such a schema verbatim
    # makes the API reject the request outright ("Input should be
    # 'TYPE_UNSPECIFIED', ..."). Found by actually running checklist
    # generation against the live Gemini API with a real key — every
    # request failed before any model call happened. The normalization
    # below is Gemini-specific, so it belongs here in the adapter, not
    # in the shared schema (which Claude/OpenAI accept as-is).
    @staticmethod
    def _sanitize_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "type":
                mapped, nullable = GeminiAdapter._gemini_type(value)
                out[key] = mapped
                if nullable:
                    out["nullable"] = True
            elif key in ("properties", "definitions") and isinstance(value, dict):
                out[key] = {
                    k: GeminiAdapter._sanitize_schema_for_gemini(v) for k, v in value.items()
                }
            elif key == "items" and isinstance(value, dict):
                out[key] = GeminiAdapter._sanitize_schema_for_gemini(value)
            else:
                out[key] = value
        return out

    @staticmethod
    def _gemini_type(type_value: Any) -> tuple[str, bool]:
        """Maps a JSON-Schema type onto Gemini's uppercase enum. Returns
        (type_enum, nullable) — `nullable` is True when a type LIST like
        ["integer", "null"] mixes a single concrete type with "null"
        (Gemini rejects the list form outright and expects
        `nullable: true` instead)."""
        _UPPER = {
            "string": "STRING",
            "integer": "INTEGER",
            "number": "NUMBER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
            "null": "NULL",
        }

        if isinstance(type_value, list):
            non_null = [t for t in type_value if t != "null"]
            if not non_null:
                return "NULL", False
            if len(non_null) == 1:
                return _UPPER.get(non_null[0], "STRING"), "null" in type_value
            # Union of several non-null types: Gemini can't express it,
            # degrade to a permissive string rather than fail the call.
            return "STRING", False
        return _UPPER.get(type_value, "STRING"), False

    @staticmethod
    def _to_gemini_contents(messages: list[Message]) -> tuple[Optional[str], list[dict]]:
        system_parts = [m.content for m in messages if m.role == "system"]
        role_map = {"user": "user", "assistant": "model", "tool": "user"}
        out = [
            {"role": role_map[m.role], "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        return ("\n".join(system_parts) or None), out

    @staticmethod
    def _to_gemini_tool(tool: ToolDefinition) -> dict:
        return {
            "function_declarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    # Same dialect problem as response_schema: tool
                    # parameter schemas (e.g. chat's
                    # CHECKLIST_STATUS_TOOL with ["integer", "null"])
                    # must be normalized for Gemini or the API rejects
                    # the request before any model call happens.
                    "parameters": GeminiAdapter._sanitize_schema_for_gemini(tool.parameters),
                }
            ]
        }

    def _from_gemini_response(
        self, resp: Any, wants_structured: bool, model: str
    ) -> LLMResponse:
        text = getattr(resp, "text", None)
        tool_calls: list[ToolCall] = []
        try:
            parts = resp.candidates[0].content.parts
            for i, part in enumerate(parts):
                fc = getattr(part, "function_call", None)
                if fc:
                    tool_calls.append(
                        ToolCall(id=f"gemini-call-{i}", name=fc.name, arguments=dict(fc.args))
                    )
        except (AttributeError, IndexError):
            pass

        structured = None
        if wants_structured and text:
            try:
                structured = json.loads(text)
            except json.JSONDecodeError:
                structured = None

        usage = getattr(resp, "usage_metadata", None)
        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            structured_output=structured,
            provider=self.name,
            model=model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            raw=resp,
        )