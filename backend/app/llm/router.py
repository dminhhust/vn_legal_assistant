"""The Model Router: the ONLY thing agent/tool code should call for LLM
completions. Never import a provider adapter or SDK directly outside
this module and its adapters.

    from app.llm.router import get_router
    router = get_router()
    response = router.complete(messages, task="legal_extraction")

Design (see docs/ARCHITECTURE.md §4.9):
  1. Detects which providers have a usable API key at startup/first use.
  2. Reads a per-task priority policy (config, not hardcoded branching —
     see task_policy.py).
  3. Tries providers in priority order; on failure, falls back to the
     next available one; raises only if every candidate fails.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Any, Optional

from app.llm.adapters.base import ProviderAdapter
from app.llm.schemas import LLMResponse, Message, ToolDefinition
from app.llm.schemas import ProviderError
from app.llm.task_policy import DEFAULT_TASK_POLICY

logger = logging.getLogger(__name__)

_ADAPTER_MODULES = {
    "claude": ("app.llm.adapters.claude_adapter", "ClaudeAdapter"),
    "openai": ("app.llm.adapters.openai_adapter", "OpenAIAdapter"),
    "gemini": ("app.llm.adapters.gemini_adapter", "GeminiAdapter"),
}

_ENV_KEY_BY_PROVIDER = {
    "claude": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}


class NoProviderAvailableError(Exception):
    """Raised when no configured provider could complete a request —
    either none have a key set, or every available one failed."""


class ModelRouter:
    def __init__(self, task_policy: Optional[dict[str, list[str]]] = None):
        self.task_policy = task_policy or DEFAULT_TASK_POLICY
        self._adapters: dict[str, ProviderAdapter] = {}
        self._available_providers: list[str] = self._detect_available_providers()

        if not self._available_providers:
            logger.warning(
                "No LLM provider API keys detected (checked %s). Set at "
                "least one of ANTHROPIC_API_KEY / OPENAI_API_KEY / "
                "GOOGLE_API_KEY.",
                list(_ENV_KEY_BY_PROVIDER.values()),
            )
        else:
            logger.info("Model Router available providers: %s", self._available_providers)

    @staticmethod
    def _detect_available_providers() -> list[str]:
        return [name for name, env_key in _ENV_KEY_BY_PROVIDER.items() if os.getenv(env_key)]

    def available_providers(self) -> list[str]:
        return list(self._available_providers)

    def _get_adapter(self, provider: str) -> ProviderAdapter:
        if provider not in self._adapters:
            module_path, class_name = _ADAPTER_MODULES[provider]
            module = importlib.import_module(module_path)
            adapter_cls = getattr(module, class_name)
            self._adapters[provider] = adapter_cls()
        return self._adapters[provider]

    def _priority_for_task(self, task: str) -> list[str]:
        order = self.task_policy.get(task, self.task_policy.get("default", []))
        # Only ever try providers that are actually configured, in the
        # order the task policy prefers.
        return [p for p in order if p in self._available_providers]

    def complete(
        self,
        messages: list[Message],
        *,
        tools: Optional[list[ToolDefinition]] = None,
        response_schema: Optional[dict[str, Any]] = None,
        task: str = "default",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResponse:
        candidates = self._priority_for_task(task)
        if not candidates:
            raise NoProviderAvailableError(
                f"No available provider for task='{task}'. "
                f"Available providers: {self._available_providers or 'none'}."
            )

        last_error: Optional[Exception] = None
        for provider in candidates:
            try:
                adapter = self._get_adapter(provider)
                return adapter.complete(
                    messages,
                    tools=tools,
                    response_schema=response_schema,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except ProviderError as exc:
                logger.warning("Provider '%s' failed for task='%s': %s", provider, task, exc)
                last_error = exc
                continue  # fall back to the next candidate
            except Exception as exc:  # noqa: BLE001 - unexpected adapter bug; still fall back
                logger.exception("Unexpected error from provider '%s'", provider)
                last_error = exc
                continue

        raise NoProviderAvailableError(
            f"All available providers failed for task='{task}': {candidates}. "
            f"Last error: {last_error}"
        )


_router_singleton: Optional[ModelRouter] = None


def get_router() -> ModelRouter:
    """Process-wide singleton so adapters (and their SDK clients) aren't
    re-created on every call."""
    global _router_singleton
    if _router_singleton is None:
        _router_singleton = ModelRouter()
    return _router_singleton
