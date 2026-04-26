from __future__ import annotations

import math
import re
from typing import Any

_TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "why",
    "with",
    "与",
    "了",
    "在",
    "是",
    "的",
}


def normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def keyword_recall(expected_keywords: list[str], answer: str | None) -> tuple[float | None, list[str]]:
    if not expected_keywords:
        return None, []
    answer_text = normalize_text(answer)
    matched = [keyword for keyword in expected_keywords if normalize_text(keyword) in answer_text]
    return len(matched) / len(expected_keywords), matched


def retrieval_recall_at_k(
    *,
    expected_evidence_ids: list[str],
    expected_source_ids: list[str],
    expected_retrieval_keywords: list[str],
    hit_payloads: list[dict[str, Any]],
    k: int = 5,
) -> tuple[bool | None, float | None, list[str], list[str], list[str]]:
    top_hits = hit_payloads[:k]
    if expected_evidence_ids:
        hit_evidence_ids = {
            evidence_id
            for hit in top_hits
            for evidence_id in hit.get("evidence_ids", [])
        }
        matched_ids = [item for item in expected_evidence_ids if item in hit_evidence_ids]
        return bool(matched_ids), len(matched_ids) / len(expected_evidence_ids), matched_ids, [], []
    if expected_source_ids:
        hit_source_ids = {
            identifier
            for hit in top_hits
            for identifier in _hit_source_identifiers(hit)
        }
        expected = _unique_identifiers(expected_source_ids)
        if not expected:
            return None, None, [], [], []
        matched_ids = [item for item in expected if item in hit_source_ids]
        return bool(matched_ids), len(matched_ids) / len(expected), [], matched_ids, []
    if expected_retrieval_keywords:
        matched_keywords = []
        haystacks = [normalize_text(hit.get("content")) for hit in top_hits]
        for keyword in expected_retrieval_keywords:
            normalized = normalize_text(keyword)
            if any(normalized in haystack for haystack in haystacks):
                matched_keywords.append(keyword)
        return bool(matched_keywords), len(matched_keywords) / len(expected_retrieval_keywords), [], [], matched_keywords
    return None, None, [], [], []


def _hit_source_identifiers(hit: dict[str, Any]) -> list[str]:
    identifiers = [
        hit.get("id"),
        hit.get("source_id"),
        hit.get("document_id"),
    ]
    identifiers.extend(hit.get("evidence_ids") or [])
    return [
        normalized
        for item in identifiers
        if (normalized := _normalize_identifier(item))
    ]


def _normalize_identifier(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _unique_identifiers(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_identifier(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def groundedness_score(
    *,
    answer: str | None,
    evidence_texts: list[str],
    grounding_keywords: list[str] | None = None,
) -> float | None:
    answer_text = normalize_text(answer)
    normalized_evidence = "\n".join(normalize_text(text) for text in evidence_texts if normalize_text(text))
    if not answer_text or not normalized_evidence:
        return None

    keywords = [normalize_text(keyword) for keyword in (grounding_keywords or []) if normalize_text(keyword)]
    if keywords:
        answer_keywords = [keyword for keyword in keywords if keyword in answer_text]
        if not answer_keywords:
            return 0.0
        supported = [keyword for keyword in answer_keywords if keyword in normalized_evidence]
        return len(supported) / len(answer_keywords)

    tokens = informative_tokens(answer_text)
    if not tokens:
        return 0.0
    supported = [token for token in tokens if token in normalized_evidence]
    return len(supported) / len(tokens)


def informative_tokens(text: str) -> list[str]:
    unique_tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(text):
        normalized = token.lower()
        if len(normalized) <= 1 or normalized in _STOPWORDS:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_tokens.append(normalized)
    return unique_tokens


def route_accuracy(expected_route: str | None, actual_route: str) -> bool | None:
    if not expected_route:
        return None
    return normalize_text(expected_route) == normalize_text(actual_route)


def reference_token_f1(reference_answer: str | None, answer: str | None) -> tuple[float | None, float | None, float | None]:
    reference_tokens = set(informative_tokens(normalize_text(reference_answer)))
    answer_tokens = set(informative_tokens(normalize_text(answer)))
    if not reference_tokens or not answer_tokens:
        return None, None, None
    overlap = reference_tokens & answer_tokens
    if not overlap:
        return 0.0, 0.0, 0.0
    precision = len(overlap) / len(answer_tokens)
    recall = len(overlap) / len(reference_tokens)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def answer_polarity(text: str | None) -> str | None:
    normalized = normalize_text(text)
    if not normalized:
        return None
    token_list = informative_tokens(normalized)
    tokens = set(token_list)
    if "yes" in tokens or normalized.startswith("yes"):
        return "yes"
    if "no" in tokens or normalized.startswith("no"):
        return "no"
    uncertain_markers = {
        "maybe",
        "unclear",
        "inconclusive",
        "insufficient",
        "unknown",
        "uncertain",
        "not",
    }
    if tokens & uncertain_markers:
        return "uncertain"
    if "证据不足" in normalized:
        return "uncertain"
    return None


def polarity_accuracy(reference_answer: str | None, answer: str | None) -> bool | None:
    expected = answer_polarity(reference_answer)
    actual = answer_polarity(answer)
    if expected is None or actual is None:
        return None
    return expected == actual


def tool_call_success_rate(
    tool_traces: list[dict[str, Any]],
    expected_tool_names: list[str] | None = None,
) -> tuple[float | None, int, int]:
    filtered = [
        trace
        for trace in tool_traces
        if trace.get("status") not in {"started", "skipped"}
    ]
    if expected_tool_names:
        expected = {name for name in expected_tool_names}
        filtered = [trace for trace in filtered if trace.get("tool_name") in expected]
    if not filtered:
        return None, 0, 0
    succeeded = sum(1 for trace in filtered if trace.get("status") == "succeeded")
    total = len(filtered)
    return succeeded / total, succeeded, total


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = max(0, min(len(ordered) - 1, math.ceil(ratio * len(ordered)) - 1))
    return ordered[rank]
