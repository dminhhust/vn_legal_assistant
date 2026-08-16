"""Unified data shapes used by every LLM provider adapter.

Agent code should only ever import from here + router.py — never a
provider SDK directly. That separation is what makes swapping or adding
providers safe (see docs/ARCHITECTURE.md §4.9).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    # populated when role == "tool": which tool call this message answers
    tool_call_id: Optional[str] = None


@dataclass
class ToolDefinition:
    """Provider-agnostic tool/function definition (JSON-schema style)."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    text: Optional[str]
    tool_calls: list[ToolCall] = field(default_factory=list)
    structured_output: Optional[dict[str, Any]] = None
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    raw: Any = None  # original provider response object, for debugging only


class ProviderError(Exception):
    """Raised by an adapter on any failure (auth, rate limit, timeout,
    malformed response). The Router catches this specifically and falls
    back to the next available provider."""

    def __init__(self, provider: str, message: str, retryable: bool = True):
        self.provider = provider
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")
