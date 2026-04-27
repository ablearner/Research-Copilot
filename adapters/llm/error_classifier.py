from __future__ import annotations

"""Structured error classification for LLM API failures.

Replaces ad-hoc status code checks with a unified ClassifiedError
that carries recovery hints (retryable, should_compress, should_fallback).
"""

from dataclasses import dataclass
from enum import Enum


class FailureReason(Enum):
    rate_limit = "rate_limit"
    billing = "billing"
    auth = "auth"
    context_overflow = "context_overflow"
    content_filter = "content_filter"
    server_error = "server_error"
    timeout = "timeout"
    connection = "connection"
    unknown = "unknown"


@dataclass(frozen=True)
class ClassifiedError:
    reason: FailureReason
    status_code: int | None
    retryable: bool
    should_compress: bool
    should_fallback: bool


def classify_llm_error(
    exc: Exception,
    *,
    provider: str = "",
) -> ClassifiedError:
    """Classify an LLM API error into a structured recovery hint."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()

    # Context overflow — most actionable
    _CTX_OVERFLOW_KEYWORDS = (
        "context_length",
        "max_tokens",
        "too long",
        "token limit",
        "maximum context",
        "context window",
        "request too large",
    )
    if status == 413 or (
        status in (400, None)
        and any(k in msg for k in _CTX_OVERFLOW_KEYWORDS)
    ):
        return ClassifiedError(
            FailureReason.context_overflow, status,
            retryable=False, should_compress=True, should_fallback=False,
        )

    # Rate limit
    if status == 429 or "rate" in msg and "limit" in msg:
        return ClassifiedError(
            FailureReason.rate_limit, status,
            retryable=True, should_compress=False, should_fallback=True,
        )

    # Billing / quota
    if status == 402 or any(k in msg for k in ("quota", "billing", "insufficient_quota", "allocationquota")):
        return ClassifiedError(
            FailureReason.billing, status,
            retryable=False, should_compress=False, should_fallback=True,
        )

    # Auth
    if status in (401, 403) or any(
        k in msg for k in ("authentication", "unauthorized", "forbidden", "permission denied")
    ):
        return ClassifiedError(
            FailureReason.auth, status,
            retryable=False, should_compress=False, should_fallback=True,
        )

    # Content filter / safety
    if any(k in msg for k in ("content_filter", "content_policy", "safety", "moderation")):
        return ClassifiedError(
            FailureReason.content_filter, status,
            retryable=False, should_compress=False, should_fallback=False,
        )

    # Server error (5xx)
    if isinstance(status, int) and 500 <= status < 600:
        return ClassifiedError(
            FailureReason.server_error, status,
            retryable=True, should_compress=False, should_fallback=True,
        )

    # Timeout / connection
    if isinstance(exc, (TimeoutError, OSError)) or "timeout" in msg:
        reason = FailureReason.timeout if ("timeout" in msg or isinstance(exc, TimeoutError)) else FailureReason.connection
        return ClassifiedError(
            reason, None,
            retryable=True, should_compress=False, should_fallback=True,
        )

    return ClassifiedError(
        FailureReason.unknown, status,
        retryable=True, should_compress=False, should_fallback=False,
    )
