"""Per-task provider priority policy.

Intentionally plain Python data, not branching code, per the
architecture doc's "provider selection policy is config, not hardcoded"
principle (docs/ARCHITECTURE.md §4.9). Edit this dict (or later load it
from YAML/DB) to change routing behavior without touching agent or
router code.

Only providers with a configured API key are ever actually tried — see
ModelRouter._priority_for_task in router.py — and "default" is the
fallback for any task name not listed here.
"""

DEFAULT_TASK_POLICY: dict[str, list[str]] = {
    # High-stakes structured extraction (e.g. legal obligation extraction):
    # prefer the strongest reasoning first.
    "legal_extraction": ["claude", "openai", "gemini"],
    # Lower-stakes, latency-sensitive writing: prefer fast/cheap first.
    "daily_briefing_writing": ["gemini", "openai", "claude"],
    "news_summarization": ["gemini", "openai", "claude"],
    # General chat / supervisor routing: balanced default order.
    "chat": ["claude", "openai", "gemini"],
    "intent_classification": ["openai", "gemini", "claude"],
    # Memory reflection (deciding what's worth remembering long-term):
    "memory_reflection": ["claude", "gemini", "openai"],
    # Fallback for anything not explicitly listed above.
    "default": ["claude", "openai", "gemini"],
}
