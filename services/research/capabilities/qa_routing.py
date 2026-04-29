from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_CHART_QUESTION_MARKERS = ("图", "图表", "figure", "fig", "chart", "plot", "diagram", "axis")
_COLLECTION_QUESTION_MARKERS = ("这些论文", "这组论文", "整体", "总体", "对比", "比较", "综述", "总结", "papers", "collection", "overall", "across papers", "compare", "comparison")
_RECOMMENDATION_OR_SELECTION_MARKERS = ("哪篇", "优先阅读", "推荐", "先读", "阅读建议", "which paper", "best paper", "worth reading", "read first")
_DOCUMENT_DRILLDOWN_MARKERS = ("这篇论文", "该论文", "原文", "正文", "段落", "页", "section", "paragraph", "page", "document", "single paper")


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def _looks_like_chart_question(normalized: str) -> bool:
    return _contains_any(normalized, _CHART_QUESTION_MARKERS)


def _looks_like_collection_question(normalized: str) -> bool:
    return _contains_any(normalized, _COLLECTION_QUESTION_MARKERS)


def _looks_like_recommendation_question(normalized: str) -> bool:
    return _contains_any(normalized, _RECOMMENDATION_OR_SELECTION_MARKERS)


def _looks_like_document_drilldown(normalized: str) -> bool:
    return _contains_any(normalized, _DOCUMENT_DRILLDOWN_MARKERS)

_QA_ROUTING_PROMPT = (
    "你是科研助手中的问答路由技能。请根据用户问题和当前 scope，判断最合适的 QA route。\n\n"
    "用户问题：{question}\n"
    "scope_mode：{scope_mode}\n"
    "selected_paper_count：{paper_count}\n"
    "selected_document_count：{document_count}\n"
    "has_explicit_visual_anchor：{has_visual_anchor}\n\n"
    "可选 route:\n"
    "- collection_qa: 面向论文集合的综合问答\n"
    "- document_drilldown: 面向单篇论文正文/段落/章节的问答\n"
    "- chart_drilldown: 面向单篇论文中的图、表、结构图、架构图、流程图、figure 的问答\n\n"
    "要求：\n"
    "- 只返回一个 route\n"
    "- 给出简短 rationale\n"
    "- 给出 0 到 1 的 confidence\n"
    "- 如果问题在问图里有什么、图怎么画、系统结构图长什么样、Figure X 表达什么，优先 chart_drilldown\n"
    "- 如果问题在问“哪篇更值得读、哪篇最适合回答、应该先看哪篇”，即使 scope 已经缩小，也仍优先 collection_qa\n"
    "- 显式输入约束优先于词面 marker，但不要因为少数关键词就忽略整体语义\n"
)


class ResearchQARouteResult(BaseModel):
    route: Literal["collection_qa", "document_drilldown", "chart_drilldown"]
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    source: Literal["llm", "heuristic"] = "heuristic"


class ResearchQARouter:
    name = "ResearchQARouter"

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    async def classify_async(
        self,
        *,
        question: str,
        scope_mode: str,
        paper_ids: list[str],
        document_ids: list[str],
        has_visual_anchor: bool,
    ) -> ResearchQARouteResult:
        normalized = " ".join(question.strip().lower().split())
        structured_override = self._structured_override(
            scope_mode=scope_mode,
            paper_ids=paper_ids,
            document_ids=document_ids,
            has_visual_anchor=has_visual_anchor,
        )
        if structured_override is not None:
            return structured_override
        if self.llm_adapter is not None:
            try:
                result = await self.llm_adapter.generate_structured(
                    prompt=_QA_ROUTING_PROMPT,
                    input_data={
                        "question": question.strip(),
                        "scope_mode": scope_mode,
                        "paper_count": str(len(paper_ids)),
                        "document_count": str(len(document_ids)),
                        "has_visual_anchor": str(bool(has_visual_anchor)).lower(),
                        "heuristic_hint": self._heuristic_classify(
                            normalized=normalized,
                            scope_mode=scope_mode,
                            paper_ids=paper_ids,
                            document_ids=document_ids,
                            has_visual_anchor=False,
                        ).model_dump(mode="json"),
                    },
                    response_model=ResearchQARouteResult,
                )
                return result.model_copy(update={"source": "llm"})
            except Exception as exc:  # noqa: BLE001
                logger.warning("QA routing skill fell back to heuristic routing: %s", exc)
        return self._heuristic_classify(
            normalized=normalized,
            scope_mode=scope_mode,
            paper_ids=paper_ids,
            document_ids=document_ids,
            has_visual_anchor=has_visual_anchor,
        )

    def _structured_override(
        self,
        *,
        scope_mode: str,
        paper_ids: list[str],
        document_ids: list[str],
        has_visual_anchor: bool,
    ) -> ResearchQARouteResult | None:
        if has_visual_anchor:
            return ResearchQARouteResult(
                route="chart_drilldown",
                confidence=0.95,
                rationale="A structured visual anchor was already resolved, so chart drilldown is the correct route.",
                source="heuristic",
            )
        return None

    def _heuristic_classify(
        self,
        *,
        normalized: str,
        scope_mode: str,
        paper_ids: list[str],
        document_ids: list[str],
        has_visual_anchor: bool,
    ) -> ResearchQARouteResult:
        if not normalized:
            return ResearchQARouteResult(
                route="collection_qa",
                confidence=0.35,
                rationale="Question text is empty after normalization, so default to collection QA.",
                source="heuristic",
            )
        if _looks_like_collection_question(normalized):
            return ResearchQARouteResult(
                route="collection_qa",
                confidence=0.9,
                rationale="Question contains collection-level comparison or synthesis markers.",
                source="heuristic",
            )
        if _looks_like_recommendation_question(normalized):
            return ResearchQARouteResult(
                route="collection_qa",
                confidence=0.86,
                rationale="Question asks for paper selection or reading priority, which is still a collection-level judgment even under a narrowed scope.",
                source="heuristic",
            )
        if len(document_ids) == 1 or len(paper_ids) == 1 or scope_mode in {"selected_documents", "selected_papers"}:
            if _looks_like_chart_question(normalized):
                return ResearchQARouteResult(
                    route="chart_drilldown",
                    confidence=0.87,
                    rationale="Question targets a single scoped paper/document and contains chart-oriented markers.",
                    source="heuristic",
                )
            if _looks_like_document_drilldown(normalized):
                return ResearchQARouteResult(
                    route="document_drilldown",
                    confidence=0.82,
                    rationale="Question targets a single scoped paper/document and contains document detail markers.",
                    source="heuristic",
                )
            if scope_mode in {"selected_documents", "selected_papers"}:
                return ResearchQARouteResult(
                    route="document_drilldown",
                    confidence=0.6,
                    rationale="QA scope is narrowed, so document drilldown is a reasonable fallback when no stronger semantic signal is available.",
                    source="heuristic",
                )
        return ResearchQARouteResult(
            route="collection_qa",
            confidence=0.62,
            rationale="No strong single-document or chart signals were detected, so collection QA remains the default route.",
            source="heuristic",
        )
