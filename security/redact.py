"""Sensitive data redaction for Kepler.

Strips API keys, tokens, passwords, and connection strings from text
before sending to auxiliary models (summarization) or persisting to memory.
"""

from __future__ import annotations

import re

_REDACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # OpenAI / Anthropic style API keys
    (re.compile(r"sk-[a-zA-Z0-9]{20,}"), "[REDACTED:api_key]"),
    # GitHub tokens
    (re.compile(r"ghp_[a-zA-Z0-9]{36,}"), "[REDACTED:github_token]"),
    (re.compile(r"gho_[a-zA-Z0-9]{36,}"), "[REDACTED:github_oauth]"),
    (re.compile(r"github_pat_[a-zA-Z0-9_]{20,}"), "[REDACTED:github_pat]"),
    # AWS access key ID
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[REDACTED:aws_key]"),
    # Bearer tokens
    (re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*", re.I), "[REDACTED:bearer]"),
    # Generic key=value secrets
    (
        re.compile(
            r"(?i)((?:api[_-]?key|secret|token|password|passwd|credential|authorization)"
            r"\s*[=:]\s*)[^\s,;\"']{8,}"
        ),
        r"\1[REDACTED]",
    ),
    # Connection strings (database URIs)
    (
        re.compile(r"(?i)(?:mysql|postgres|postgresql|mongodb|redis|amqp)://[^\s\"']+"),
        "[REDACTED:connection_string]",
    ),
    # .env file content patterns
    (
        re.compile(r"(?m)^[A-Z_]{3,}=\S{8,}$"),
        "[REDACTED:env_value]",
    ),
]


def redact_sensitive_text(text: str) -> str:
    """Replace detected secrets in *text* with safe placeholders."""
    for pattern, replacement in _REDACT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
