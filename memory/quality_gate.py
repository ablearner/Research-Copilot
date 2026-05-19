from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class MemoryQualityDecision(BaseModel):
    allowed: bool
    score: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    signals: dict[str, Any] = Field(default_factory=dict)


class MemoryQualityGate:
    """Rule-based gate for long-term research conclusions.

    Long-term memory should contain stable, evidence-backed facts or user
    preferences.  Raw fallback answers, empty evidence notices, and tiny status
    strings should stay in session memory only.
    """

    _GENERIC_PATTERNS = (
        re.compile(r"证据不足|无法稳定回答|暂无法生成完整摘要|候选论文偏少"),
        re.compile(r"not enough evidence|insufficient evidence|no reliable evidence", re.I),
        re.compile(r"local smoke test|placeholder|dummy|test record", re.I),
    )

    def __init__(self, *, enabled: bool = True, min_score: float = 0.35) -> None:
        self.enabled = enabled
        self.min_score = max(0.0, min(1.0, float(min_score)))

    def evaluate_conclusion(
        self,
        *,
        content: str,
        topic: str,
        keywords: list[str] | None = None,
        related_paper_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryQualityDecision:
        if not self.enabled:
            return MemoryQualityDecision(
                allowed=True,
                score=1.0,
                reason="quality_gate_disabled",
            )

        normalized = " ".join(str(content or "").split())
        metadata = metadata or {}
        keywords = keywords or []
        related_paper_ids = related_paper_ids or []

        if len(normalized) < 24:
            return self._reject("too_short", normalized, metadata)
        if any(pattern.search(normalized) for pattern in self._GENERIC_PATTERNS):
            return self._reject("generic_or_insufficient_answer", normalized, metadata)

        evidence_count = self._int_value(metadata.get("evidence_count"))
        confidence = self._float_value(metadata.get("confidence"))
        selected_count = len([item for item in related_paper_ids if str(item).strip()])
        if selected_count <= 0:
            selected = metadata.get("selected_paper_ids") or metadata.get("paper_ids") or []
            if isinstance(selected, list):
                selected_count = len([item for item in selected if str(item).strip()])

        score = 0.20
        score += min(len(normalized) / 700.0, 0.25)
        if topic.strip():
            score += 0.12
        if keywords:
            score += min(len(keywords), 4) * 0.03
        if selected_count > 0:
            score += min(selected_count, 4) * 0.04
        if evidence_count is not None:
            score += min(evidence_count, 6) * 0.04
        if confidence is not None:
            score += max(0.0, min(confidence, 1.0)) * 0.18

        score = min(score, 1.0)
        if evidence_count == 0:
            return self._reject("no_evidence", normalized, metadata, score=score)
        if confidence is not None and confidence < 0.20:
            return self._reject("very_low_confidence", normalized, metadata, score=score)
        if score < self.min_score:
            return self._reject("quality_score_below_threshold", normalized, metadata, score=score)

        return MemoryQualityDecision(
            allowed=True,
            score=score,
            reason="accepted",
            signals={
                "content_chars": len(normalized),
                "evidence_count": evidence_count,
                "confidence": confidence,
                "related_paper_count": selected_count,
                "keyword_count": len(keywords),
            },
        )

    def _reject(
        self,
        reason: str,
        content: str,
        metadata: dict[str, Any],
        *,
        score: float = 0.0,
    ) -> MemoryQualityDecision:
        return MemoryQualityDecision(
            allowed=False,
            score=max(0.0, min(1.0, score)),
            reason=reason,
            signals={
                "content_chars": len(content),
                "evidence_count": self._int_value(metadata.get("evidence_count")),
                "confidence": self._float_value(metadata.get("confidence")),
            },
        )

    @staticmethod
    def _int_value(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _float_value(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
