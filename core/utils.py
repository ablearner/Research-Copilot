"""Lightweight shared utility functions used across multiple Kepler modules."""

from __future__ import annotations

import re
from datetime import UTC, datetime


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def normalize_topic_text(text: str | None) -> str:
    """Lowercase + strip non-alphanumeric (keep CJK) for fuzzy topic matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", str(text or "").lower())).strip()


def normalize_paper_title(title: str) -> str:
    """Lowercase + strip non-alphanumeric for paper title deduplication."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", title.lower())).strip()
