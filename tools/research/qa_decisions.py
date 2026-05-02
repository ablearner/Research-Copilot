"""Pure decision functions for QA routing and quality evaluation.

These functions contain no I/O or service state — they are purely
deterministic logic extracted from QARoutingMixin so that agents and
executors can call them without depending on the service layer.
"""

from __future__ import annotations

import re
from typing import Any

from domain.schemas.api import QAResponse
from domain.schemas.research import (
    PaperCandidate,
    ResearchTask,
    ResearchTaskAskRequest,
)
from tools.research.qa_schemas import ResearchQARouteDecision

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.research.paper_selector import PaperSelectionScope


def rewrite_collection_question(
    *,
    question: str,
    task: ResearchTask,
    papers: list[PaperCandidate],
    scope_mode: str,
) -> str:
    normalized = str(question or "").strip()
    if not normalized:
        return normalized
    compact = re.sub(r"\s+", "", normalized.lower())
    if compact in {"效果怎么样", "效果如何", "表现怎么样", "表现如何"}:
        return (
            f"请结合研究主题\u201c{task.topic}\u201d对当前研究集合做综合评价，"
            "说明整体效果、证据强弱与主要边界，不要只回答单篇论文。"
        )
    if scope_mode == "all_imported":
        return normalized
    return normalized


def select_recovery_qa_route(
    *,
    request: ResearchTaskAskRequest,
    scope: PaperSelectionScope,
    document_ids: list[str],
    qa: QAResponse,
    qa_route_decision: ResearchQARouteDecision,
    quality_check: dict[str, Any],
) -> ResearchQARouteDecision | None:
    if qa_route_decision.recovery_count >= 1:
        return None
    if qa_route_decision.visual_anchor is not None:
        return None
    if not quality_check.get("needs_recovery"):
        return None
    if document_ids and qa_route_decision.route == "collection_qa" and scope.scope_mode in {"selected_documents", "selected_papers"}:
        return ResearchQARouteDecision(
            route="document_drilldown",
            confidence=max(qa_route_decision.confidence, 0.72),
            rationale=(
                "The initial collection QA answer was under-supported for a narrowed paper/document scope, "
                "so a single conservative retry uses document drilldown."
            ),
            visual_anchor=None,
            recovery_count=qa_route_decision.recovery_count + 1,
        )
    if qa_route_decision.route == "document_drilldown" and not document_ids:
        return ResearchQARouteDecision(
            route="collection_qa",
            confidence=max(qa_route_decision.confidence, 0.7),
            rationale=(
                "The initial document drilldown route had no usable document scope, "
                "so a single conservative retry broadens to collection QA."
            ),
            visual_anchor=None,
            recovery_count=qa_route_decision.recovery_count + 1,
        )
    return None


def is_insufficient_answer(*, answer: str, confidence: float, evidence_count: int) -> bool:
    lowered = answer.lower()
    insufficient_markers = (
        "证据不足",
        "无法确认",
        "不能确认",
        "信息不足",
        "insufficient evidence",
        "not enough evidence",
    )
    return confidence < 0.45 or evidence_count < 2 or any(marker in lowered for marker in insufficient_markers)


def build_answer_quality_check(
    *,
    qa: QAResponse,
    route: str,
    scope_mode: str,
    document_ids: list[str],
) -> dict[str, Any]:
    evidence_count = len(qa.evidence_bundle.evidences)
    confidence = qa.confidence if qa.confidence is not None else 0.0
    insufficient = is_insufficient_answer(
        answer=qa.answer,
        confidence=confidence,
        evidence_count=evidence_count,
    )
    warnings: list[str] = []
    if evidence_count < 2:
        warnings.append("low_evidence_count")
    if confidence < 0.45:
        warnings.append("low_confidence")
    if route in {"document_drilldown", "chart_drilldown"} and not document_ids:
        warnings.append("drilldown_without_document_scope")
    if "无法" in qa.answer or "不能确认" in qa.answer:
        warnings.append("answer_contains_uncertainty_marker")
    return {
        "evidence_count": evidence_count,
        "confidence": round(confidence, 4),
        "route": route,
        "scope_mode": scope_mode,
        "needs_recovery": insufficient,
        "recommended_recovery": (
            "import_or_expand_evidence"
            if insufficient
            else "none"
        ),
        "warnings": warnings,
    }
