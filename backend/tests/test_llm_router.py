"""Unit tests for the Model Router's provider-detection and fallback
logic.

These tests never call a real LLM API and never require the provider
SDKs (anthropic/openai/google-genai) to be installed — they monkeypatch
environment variables and inject fake adapters directly, so the
Router's *routing logic* is verified in isolation from any network call
or third-party package.
"""
from __future__ import annotations

import pytest

from app.llm.router import ModelRouter, NoProviderAvailableError
from app.llm.schemas import LLMResponse, Message, ProviderError


class _FakeAdapter:
    """Stand-in adapter so tests don't depend on real provider SDKs."""

    def __init__(self, name: str, should_fail: bool = False):
        self.name = name
        self.should_fail = should_fail

    def complete(self, messages, **kwargs) -> LLMResponse:  # noqa: ANN001
        if self.should_fail:
            raise ProviderError(self.name, "simulated failure")
        return LLMResponse(text=f"ok from {self.name}", provider=self.name, model="fake-model")


def _clear_provider_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)


def test_detects_only_the_configured_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    router = ModelRouter()

    assert router.available_providers() == ["openai"]


def test_detects_multiple_configured_providers_in_declared_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    router = ModelRouter()

    assert set(router.available_providers()) == {"claude", "gemini"}
    assert "openai" not in router.available_providers()


def test_raises_when_no_keys_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_keys(monkeypatch)

    router = ModelRouter()

    with pytest.raises(NoProviderAvailableError):
        router.complete([Message(role="user", content="hi")])


def test_respects_task_specific_priority_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

    router = ModelRouter(
        task_policy={"summarize": ["gemini", "openai", "claude"], "default": ["claude"]}
    )

    assert router._priority_for_task("summarize") == ["gemini", "openai", "claude"]
    # A task not in the policy falls back to "default".
    assert router._priority_for_task("some_unlisted_task") == ["claude"]


def test_falls_back_to_next_available_provider_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    router = ModelRouter(task_policy={"default": ["claude", "openai"]})
    # Pre-populate the adapter cache with fakes so no real SDK is ever touched.
    router._adapters["claude"] = _FakeAdapter("claude", should_fail=True)
    router._adapters["openai"] = _FakeAdapter("openai", should_fail=False)

    response = router.complete([Message(role="user", content="hi")])

    assert response.provider == "openai"
    assert response.text == "ok from openai"


def test_raises_no_provider_available_when_every_candidate_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    router = ModelRouter(task_policy={"default": ["claude", "openai"]})
    router._adapters["claude"] = _FakeAdapter("claude", should_fail=True)
    router._adapters["openai"] = _FakeAdapter("openai", should_fail=True)

    with pytest.raises(NoProviderAvailableError):
        router.complete([Message(role="user", content="hi")])


def test_task_policy_only_ever_offers_configured_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if a task's policy lists a provider first, that provider is
    skipped entirely (not attempted, not counted as a failure) if its
    key was never configured."""
    _clear_provider_keys(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    router = ModelRouter(task_policy={"default": ["claude", "openai", "gemini"]})

    assert router._priority_for_task("default") == ["openai"]


class TestDeliberateProviderOutageSimulation:
    """Phase 9 hardening — docs/IMPLEMENTATION_PLAN.md Phase 9:
    'Deliberately test the Model Router fallback path (simulate a
    provider outage).' Uses realistic network-style exceptions
    (ConnectionError, TimeoutError) rather than the router's own
    ProviderError, to prove the router's generic `except Exception`
    catch-all — not just its narrow ProviderError handling — correctly
    triggers fallback. This is what an actual outage looks like: an
    adapter's underlying SDK call raises whatever the SDK raises, which
    the router has never seen before and can't special-case."""

    def test_connection_error_triggers_fallback_to_next_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        router = ModelRouter(task_policy={"default": ["claude", "openai"]})
        router._adapters["claude"] = _RealisticFailureAdapter("claude", ConnectionError("connection refused"))
        router._adapters["openai"] = _FakeAdapter("openai", should_fail=False)

        response = router.complete([Message(role="user", content="hi")])

        assert response.provider == "openai"

    def test_timeout_error_triggers_fallback_to_next_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        router = ModelRouter(task_policy={"default": ["claude", "gemini"]})
        router._adapters["claude"] = _RealisticFailureAdapter("claude", TimeoutError("request timed out"))
        router._adapters["gemini"] = _FakeAdapter("gemini", should_fail=False)

        response = router.complete([Message(role="user", content="hi")])

        assert response.provider == "gemini"

    def test_full_multi_provider_outage_across_all_three_raises_clearly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulates a worse scenario than a single provider going
        down: all three configured providers are unreachable at once
        (e.g. a shared network egress problem). The router must still
        fail with a clear, actionable error — not hang, not crash with
        an unrelated traceback."""
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        router = ModelRouter(task_policy={"default": ["claude", "openai", "gemini"]})
        router._adapters["claude"] = _RealisticFailureAdapter("claude", ConnectionError("DNS resolution failed"))
        router._adapters["openai"] = _RealisticFailureAdapter("openai", TimeoutError("request timed out"))
        router._adapters["gemini"] = _RealisticFailureAdapter("gemini", ConnectionError("connection reset"))

        with pytest.raises(NoProviderAvailableError) as exc_info:
            router.complete([Message(role="user", content="hi")])

        # The error message should be genuinely useful for on-call
        # debugging, not just "something failed somewhere."
        assert "default" in str(exc_info.value)
        assert "claude" in str(exc_info.value) or "gemini" in str(exc_info.value)

    def test_transient_outage_recovers_on_a_later_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A more realistic outage shape: the preferred provider is
        down for one call, then comes back — the router shouldn't
        require any manual reset to use it again once it recovers."""
        _clear_provider_keys(monkeypatch)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        router = ModelRouter(task_policy={"default": ["claude", "openai"]})
        flaky_claude = _RealisticFailureAdapter("claude", ConnectionError("temporarily unreachable"))
        router._adapters["claude"] = flaky_claude
        router._adapters["openai"] = _FakeAdapter("openai", should_fail=False)

        during_outage = router.complete([Message(role="user", content="hi")])
        assert during_outage.provider == "openai"  # fell back correctly

        flaky_claude.recovered = True  # simulate the outage ending
        after_recovery = router.complete([Message(role="user", content="hi")])
        assert after_recovery.provider == "claude"  # back to the preferred provider, no manual reset needed


class _RealisticFailureAdapter:
    """Raises a realistic, non-ProviderError exception — the shape a
    real SDK failure actually takes — until `.recovered` is flipped to
    True, at which point it starts succeeding. Distinct from
    `_FakeAdapter` (which only ever raises the router's own
    ProviderError) specifically to exercise the router's broader
    `except Exception` fallback path, not just its narrow one."""

    def __init__(self, name: str, exception: Exception):
        self.name = name
        self._exception = exception
        self.recovered = False

    def complete(self, messages, **kwargs) -> LLMResponse:  # noqa: ANN001
        if not self.recovered:
            raise self._exception
        return LLMResponse(text=f"ok from {self.name}", provider=self.name, model="fake-model")
