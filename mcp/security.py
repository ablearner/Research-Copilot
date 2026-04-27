"""MCP security utilities — safe env filtering, error sanitization, injection scanning."""

from __future__ import annotations

import os
import re

SAFE_ENV_KEYS: frozenset[str] = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
    "XDG_RUNTIME_DIR", "DISPLAY", "WAYLAND_DISPLAY",
})

_CREDENTIAL_PATTERN = re.compile(
    r"(?:ghp_\S+|gho_\S+|github_pat_\S+|sk-[A-Za-z0-9_-]{8,}|Bearer\s+\S+|"
    r"token=\S+|AKIA[0-9A-Z]{16})",
    re.I,
)

_MCP_INJECTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(previous|all|above)\s+instructions", re.I), "prompt_injection"),
    (re.compile(r"</?system[^>]*>", re.I), "xml_injection"),
    (re.compile(r"\[\[SYSTEM\]\]", re.I), "bracket_injection"),
    (re.compile(r"you\s+are\s+now\s+", re.I), "role_hijack"),
]


def build_safe_env(user_env: dict[str, str] | None = None) -> dict[str, str]:
    """Build a sanitized environment dict for MCP subprocess spawning.

    Only whitelisted keys from the current process env plus any
    explicitly passed *user_env* keys are included.
    """
    base = {k: v for k, v in os.environ.items() if k in SAFE_ENV_KEYS}
    if user_env:
        base.update(user_env)
    return base


def sanitize_error(text: str) -> str:
    """Strip credentials from error messages before logging or returning to user."""
    return _CREDENTIAL_PATTERN.sub("[REDACTED]", text)


def scan_tool_description(description: str) -> list[str]:
    """Check an MCP tool description for injection patterns.

    Returns a list of warning strings (empty if clean).
    """
    warnings: list[str] = []
    for pattern, label in _MCP_INJECTION_PATTERNS:
        if pattern.search(description):
            warnings.append(f"MCP tool description contains suspicious pattern: {label}")
    return warnings
