from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field


VisualIntentName = Literal[
    "current_visual_followup",
    "new_visual_search",
    "negative_feedback_continue_search",
    "ambiguous",
]

_CURRENT_VISUAL_PATTERNS = (
    r"这张图",
    r"这幅图",
    r"这个图",
    r"当前图",
    r"刚才.*图",
    r"图中",
    r"上图",
    r"this (figure|image|chart)",
    r"current (figure|image|chart)",
    r"in the (figure|image|chart)",
)
_NEW_VISUAL_SEARCH_PATTERNS = (
    r"找.*图",
    r"找到.*图",
    r"提供.*图",
    r"给我.*图",
    r"给出.*图",
    r"定位.*图",
    r"有没有.*图",
    r"find .* (figure|image|chart|diagram|histogram)",
    r"show me .* (figure|image|chart|diagram|histogram)",
    r"provide .* (figure|image|chart|diagram|histogram)",
    r"locate .* (figure|image|chart|diagram|histogram)",
)
_NEGATIVE_FEEDBACK_PATTERNS = (
    r"这张不是",
    r"不是这张",
    r"不是这个",
    r"不是这幅",
    r"找错",
    r"错图",
    r"不对",
    r"换一张",
    r"下一张",
    r"继续找",
    r"重新找",
    r"not this",
    r"wrong (figure|image|chart)",
    r"try another",
)
_VISUAL_TARGET_PATTERNS = (
    r"图",
    r"图表",
    r"直方图",
    r"流程图",
    r"框图",
    r"架构图",
    r"figure",
    r"fig\.",
    r"chart",
    r"image",
    r"diagram",
    r"histogram",
    r"plot",
)


class VisualIntentDecision(BaseModel):
    intent: VisualIntentName = "ambiguous"
    reuse_current_anchor: bool = False
    search_new_figure: bool = False
    target_description: str | None = None
    exclude_figure_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = ""
    marker_signals: dict[str, list[str]] = Field(default_factory=dict)


class VisualIntentRouter:
    """Classify whether a visual question should reuse the current image or find a new one."""

    def __init__(self, *, llm_adapter: Any | None = None) -> None:
        self.llm_adapter = llm_adapter

    async def decide_async(
        self,
        *,
        question: str,
        current_visual_anchor: dict[str, Any] | None = None,
        current_figure: dict[str, Any] | None = None,
        has_new_image: bool = False,
    ) -> VisualIntentDecision:
        signals = self.marker_signals(question)
        fallback = self._fallback_decision(
            question=question,
            current_visual_anchor=current_visual_anchor,
            current_figure=current_figure,
            has_new_image=has_new_image,
            marker_signals=signals,
        )
        if self.llm_adapter is None:
            return fallback
        try:
            result = await self.llm_adapter.generate_structured(
                prompt=_VISUAL_INTENT_PROMPT,
                input_data={
                    "question": question.strip(),
                    "has_new_image": has_new_image,
                    "current_visual_anchor": self._compact_anchor(current_visual_anchor),
                    "current_figure": self._compact_anchor(current_figure),
                    "marker_signals": signals,
                    "fallback_decision": fallback.model_dump(mode="json"),
                },
                response_model=VisualIntentDecision,
            )
        except Exception:
            return fallback
        return self._normalize_decision(
            result,
            fallback=fallback,
            current_visual_anchor=current_visual_anchor,
            current_figure=current_figure,
            marker_signals=signals,
        )

    @staticmethod
    def marker_signals(question: str) -> dict[str, list[str]]:
        return {
            "current_visual": _matches(_CURRENT_VISUAL_PATTERNS, question),
            "new_visual_search": _matches(_NEW_VISUAL_SEARCH_PATTERNS, question),
            "negative_feedback": _matches(_NEGATIVE_FEEDBACK_PATTERNS, question),
            "visual_target": _matches(_VISUAL_TARGET_PATTERNS, question),
        }

    def _fallback_decision(
        self,
        *,
        question: str,
        current_visual_anchor: dict[str, Any] | None,
        current_figure: dict[str, Any] | None,
        has_new_image: bool,
        marker_signals: dict[str, list[str]],
    ) -> VisualIntentDecision:
        current_ids = self._current_figure_ids(current_visual_anchor, current_figure)
        if has_new_image:
            return VisualIntentDecision(
                intent="current_visual_followup",
                reuse_current_anchor=True,
                search_new_figure=False,
                target_description=question.strip() or None,
                confidence=0.72,
                rationale="A new image was provided with the request, so the visual question should use that image.",
                marker_signals=marker_signals,
            )
        if marker_signals["negative_feedback"]:
            return VisualIntentDecision(
                intent="negative_feedback_continue_search",
                reuse_current_anchor=False,
                search_new_figure=True,
                target_description=question.strip() or None,
                exclude_figure_ids=current_ids,
                confidence=0.78,
                rationale="The user rejected the previous visual and asked to continue searching.",
                marker_signals=marker_signals,
            )
        if marker_signals["new_visual_search"] and marker_signals["visual_target"]:
            return VisualIntentDecision(
                intent="new_visual_search",
                reuse_current_anchor=False,
                search_new_figure=True,
                target_description=question.strip() or None,
                exclude_figure_ids=current_ids,
                confidence=0.68,
                rationale="The user appears to request locating a new visual from the research set.",
                marker_signals=marker_signals,
            )
        if marker_signals["current_visual"] and current_visual_anchor:
            return VisualIntentDecision(
                intent="current_visual_followup",
                reuse_current_anchor=True,
                search_new_figure=False,
                target_description=question.strip() or None,
                confidence=0.66,
                rationale="The user explicitly refers to the current visual.",
                marker_signals=marker_signals,
            )
        return VisualIntentDecision(
            intent="ambiguous",
            reuse_current_anchor=False,
            search_new_figure=False,
            target_description=question.strip() or None,
            confidence=0.35,
            rationale="The question does not clearly indicate whether to reuse the current visual or find a new one.",
            marker_signals=marker_signals,
        )

    def _normalize_decision(
        self,
        decision: VisualIntentDecision,
        *,
        fallback: VisualIntentDecision,
        current_visual_anchor: dict[str, Any] | None,
        current_figure: dict[str, Any] | None,
        marker_signals: dict[str, list[str]],
    ) -> VisualIntentDecision:
        if decision.intent == "ambiguous" and fallback.intent != "ambiguous" and decision.confidence < 0.5:
            decision = fallback
        excluded = list(dict.fromkeys([*decision.exclude_figure_ids, *fallback.exclude_figure_ids]))
        current_ids = self._current_figure_ids(current_visual_anchor, current_figure)
        if decision.search_new_figure:
            excluded = list(dict.fromkeys([*excluded, *current_ids]))
        intent = decision.intent
        reuse_current = decision.reuse_current_anchor
        search_new = decision.search_new_figure
        if intent == "current_visual_followup":
            reuse_current = bool(current_visual_anchor)
            search_new = False
        elif intent in {"new_visual_search", "negative_feedback_continue_search"}:
            reuse_current = False
            search_new = True
        return decision.model_copy(
            update={
                "reuse_current_anchor": reuse_current,
                "search_new_figure": search_new,
                "exclude_figure_ids": excluded,
                "target_description": decision.target_description or fallback.target_description,
                "marker_signals": marker_signals,
            }
        )

    @staticmethod
    def _compact_anchor(anchor: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(anchor, dict):
            return None
        return {
            key: anchor.get(key)
            for key in (
                "paper_id",
                "figure_id",
                "chart_id",
                "page_id",
                "page_number",
                "title",
                "caption",
                "source",
                "image_path",
            )
            if anchor.get(key) is not None
        }

    @staticmethod
    def _current_figure_ids(*anchors: dict[str, Any] | None) -> list[str]:
        ids: list[str] = []
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            figure_id = str(anchor.get("figure_id") or "").strip()
            if figure_id and figure_id not in ids:
                ids.append(figure_id)
        return ids


def _matches(patterns: tuple[str, ...], text: str) -> list[str]:
    lowered = text.lower()
    return [pattern for pattern in patterns if re.search(pattern, lowered, re.IGNORECASE)]


_VISUAL_INTENT_PROMPT = (
    "你是科研助手的视觉意图路由器。请判断用户是在继续追问当前图片，还是要求从研究集合/论文中寻找新的图片。\n\n"
    "用户问题：{question}\n"
    "是否本轮提供了新图片：{has_new_image}\n"
    "当前图片锚点：{current_visual_anchor}\n"
    "当前 figure 信息：{current_figure}\n"
    "marker 信号（仅作参考，不可作为唯一依据）：{marker_signals}\n"
    "启发式备选判断：{fallback_decision}\n\n"
    "要求：\n"
    "- marker 只作为提示，最终按整句语义判断。\n"
    "- 如果用户问“这张图/图中/刚才那张”的内容，选择 current_visual_followup。\n"
    "- 如果用户说“找/提供/给我/定位某类图”，即使当前已有图片，也通常选择 new_visual_search。\n"
    "- 如果用户明确说上一张不对、继续找、换一张，选择 negative_feedback_continue_search，并排除当前 figure_id。\n"
    "- 如果无法判断，选择 ambiguous，不要默认复用当前图片。\n"
    "- 仅返回结构化字段。"
)
