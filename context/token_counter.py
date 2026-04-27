"""Token budget management for Kepler context windows.

Provides model context length lookup, rough token estimation,
and a TokenBudget tracker that knows when compression is needed.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Model context length table ─────────────────────────────────────

MODEL_CONTEXT_LENGTHS: dict[str, int] = {
    # Qwen (DashScope)
    "qwen-plus": 131_072,
    "qwen-turbo": 131_072,
    "qwen-max": 32_768,
    "qwen-long": 10_000_000,
    "qwen3-235b-a22b": 131_072,
    "qwen3-30b-a3b": 131_072,
    "qwen2.5-72b-instruct": 131_072,
    "qwen2.5-32b-instruct": 131_072,
    # OpenAI
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
    # Anthropic
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-opus-20240229": 200_000,
    # Google
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-pro": 1_048_576,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    # DeepSeek
    "deepseek-chat": 65_536,
    "deepseek-reasoner": 65_536,
}

_DEFAULT_CONTEXT_LENGTH = 32_768


def get_context_length(model: str) -> int:
    """Return known context length for *model*, falling back to 32K."""
    if model in MODEL_CONTEXT_LENGTHS:
        return MODEL_CONTEXT_LENGTHS[model]
    model_lower = model.lower()
    for key, length in MODEL_CONTEXT_LENGTHS.items():
        if key in model_lower:
            return length
    return _DEFAULT_CONTEXT_LENGTH


def estimate_tokens_rough(messages: list[dict]) -> int:
    """Fast char/4 estimation with per-message overhead — no external deps."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(str(block.get("text", "")))
                else:
                    total_chars += len(str(block))
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function", {})
                    total_chars += len(str(fn.get("name", "")))
                    total_chars += len(str(fn.get("arguments", "")))
    return total_chars // 4 + len(messages) * 4


class TokenBudget:
    """Tracks remaining token budget for a model across an agent turn."""

    def __init__(self, model: str, *, reserve_for_output: int = 4096) -> None:
        self.model = model
        self.total = get_context_length(model)
        self.reserved_output = reserve_for_output
        self.available = self.total - reserve_for_output
        self.used = 0
        self._probed = False

    def consume(self, tokens: int, label: str = "") -> int:
        """Record *tokens* consumed and return remaining budget."""
        self.used += tokens
        return self.remaining

    @property
    def remaining(self) -> int:
        return max(0, self.available - self.used)

    def should_compress(self, threshold: float = 0.85) -> bool:
        """Return True when used ratio exceeds *threshold*."""
        if self.available <= 0:
            return False
        return self.used / self.available > threshold

    # ── Context window probing (改进 15) ──────────────────────────

    def handle_context_overflow(self, error_msg: str) -> bool:
        """React to a context-overflow error by adjusting limits.

        Returns True if the budget was adjusted (caller should retry
        after compression).  Only adjusts once per session to avoid loops.
        """
        if self._probed:
            return False

        match = re.search(r"maximum.*?(\d{4,})\s*token", error_msg, re.I)
        if match:
            actual_limit = int(match.group(1))
            if actual_limit < self.total:
                logger.warning(
                    "Context probe: model reports %d tokens (was %d), adjusting",
                    actual_limit,
                    self.total,
                )
                self.total = actual_limit
                self.available = self.total - self.reserved_output
                self._probed = True
                return True

        reduced = int(self.total * 0.75)
        logger.warning(
            "Context probe: reducing budget from %d to %d",
            self.total,
            reduced,
        )
        self.total = reduced
        self.available = self.total - self.reserved_output
        self._probed = True
        return True

    def reset_usage(self) -> None:
        """Reset consumed tokens for a new turn."""
        self.used = 0
