"""Anthropic prompt caching support for Kepler.

Places up to 4 ``cache_control`` breakpoints on messages so that
Anthropic's server-side prompt caching can kick in.  Pure function,
no state, deep-copy safe.
"""

from __future__ import annotations

import copy
from typing import Any


def apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Insert up to 4 cache_control breakpoints: system + last 3 non-system.

    Returns a **deep copy** — the original list is never mutated.
    """
    messages = copy.deepcopy(messages)
    marker = {"type": "ephemeral"}
    breakpoints = 0

    if messages and messages[0].get("role") == "system":
        _mark(messages[0], marker)
        breakpoints += 1

    remaining = 4 - breakpoints
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]
    for idx in non_sys[-remaining:]:
        _mark(messages[idx], marker)

    return messages


def _mark(msg: dict[str, Any], marker: dict[str, str]) -> None:
    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = [{"type": "text", "text": content, "cache_control": marker}]
    elif isinstance(content, list) and content:
        content[-1]["cache_control"] = marker
    else:
        msg["cache_control"] = marker
