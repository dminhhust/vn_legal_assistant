"""Abstract interface every provider adapter must implement.

One adapter per LLM provider. Each adapter translates the unified
interface (messages / tools / response_schema) into that provider's
actual API shape, and normalizes the response back into LLMResponse.

Adapters must NEVER be imported or called directly by agent code —
only the Router (router.py) should hold references to adapters. This
is what lets the rest of the codebase stay provider-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.llm.schemas import LLMResponse, Message, ToolDefinition


class ProviderAdapter(ABC):
    name: str = "base"

    @abstractmethod
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
        """Run one completion.

        Must raise app.llm.schemas.ProviderError on any failure so the
        Router can decide whether to fall back to the next provider —
        never let a raw SDK exception escape an adapter.
        """
        raise NotImplementedError
