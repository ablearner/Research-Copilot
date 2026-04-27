"""Dangerous command approval gate for Kepler tool execution.

Detects potentially destructive operations in tool arguments
and provides a callback mechanism for user confirmation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[rf]+\s+|.*\s+/)"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\s+777\b"),
    re.compile(r"\bdrop\s+table\b", re.I),
    re.compile(r"\bdelete\s+from\b", re.I),
    re.compile(r"\btruncate\b", re.I),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\s+if="),
    re.compile(r"\bformat\s+[a-z]:", re.I),
]


class ApprovalGate:
    """Pre-execution check for potentially dangerous tool operations."""

    def __init__(
        self,
        callback: Callable[[str, str, dict[str, Any]], Awaitable[bool]] | None = None,
        auto_approve_tools: set[str] | None = None,
    ) -> None:
        self.callback = callback
        self.auto_approve_tools = auto_approve_tools or set()

    async def check(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """Return True if approved, False if rejected."""
        if tool_name in self.auto_approve_tools:
            return True

        if not self._is_dangerous(tool_input):
            return True

        logger.warning(
            "Dangerous operation detected in tool '%s': %s",
            tool_name,
            {k: str(v)[:100] for k, v in tool_input.items()},
        )

        if self.callback is not None:
            return await self.callback(
                tool_name,
                "Dangerous operation detected",
                tool_input,
            )

        return False

    def _is_dangerous(self, tool_input: dict[str, Any]) -> bool:
        text = " ".join(str(v) for v in tool_input.values())
        return any(pattern.search(text) for pattern in DANGEROUS_PATTERNS)
