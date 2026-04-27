"""Memory content security scanning for Kepler.

Checks content written to long-term memory for prompt injection
patterns and invisible unicode characters that could be used for
deception or exfiltration.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

_THREAT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ignore\s+(previous|all|above|prior)\s+instructions", re.I), "prompt_injection"),
    (re.compile(r"you\s+are\s+now\s+", re.I), "role_hijack"),
    (re.compile(r"do\s+not\s+tell\s+the\s+user", re.I), "deception_hide"),
    (re.compile(r"system\s+prompt\s+override", re.I), "sys_prompt_override"),
    (re.compile(r"disregard\s+(your|all|any)\s+(instructions|rules)", re.I), "disregard_rules"),
    (re.compile(r"pretend\s+(you|to)\s+(are|be)", re.I), "pretend_role"),
    (re.compile(r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD)", re.I), "exfil_curl"),
    (re.compile(r"cat\s+[^\n]*(\.env|credentials|\.netrc)", re.I), "read_secrets"),
    (re.compile(r"</?(system|instruction|prompt)[^>]*>", re.I), "xml_injection"),
    (re.compile(r"BEGININSTRUCTION|ENDINSTRUCTION", re.I), "instruction_tags"),
    (re.compile(r"\[\[SYSTEM\]\]", re.I), "bracket_injection"),
]

_INVISIBLE_CHARS: frozenset[str] = frozenset({
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # byte order mark
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
})


def scan_memory_content(content: str) -> Optional[str]:
    """Return an error string if *content* is unsafe for memory storage.

    Returns None if content is safe.
    """
    for char in _INVISIBLE_CHARS:
        if char in content:
            return f"Blocked: invisible unicode U+{ord(char):04X}"

    for pattern, threat_id in _THREAT_PATTERNS:
        if pattern.search(content):
            return f"Blocked: threat pattern '{threat_id}'"

    return None
