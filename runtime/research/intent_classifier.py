"""Intent classification heuristics for the research runtime.

Pure functions — no runtime state needed. Extracted from
supervisor_graph_runtime_core.py to separate decision logic from execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.utils import normalize_topic_text as _normalize_topic_text_impl
from domain.schemas.research import ResearchAgentRunRequest


# ---------------------------------------------------------------------------
# Intent → route_mode mapping
# ---------------------------------------------------------------------------

_INTENT_TO_ROUTE_MODE: dict[str, str] = {
    "literature_search": "research_discovery",
    "paper_import": "paper_follow_up",
    "sync_to_zotero": "paper_follow_up",
    "collection_qa": "research_follow_up",
    "single_paper_qa": "paper_follow_up",
    "paper_comparison": "paper_follow_up",
    "paper_recommendation": "paper_follow_up",
    "figure_qa": "chart_drilldown",
    "document_understanding": "document_drilldown",
    "general_answer": "general_chat",
    "general_follow_up": "research_follow_up",
}


# ---------------------------------------------------------------------------
# Keyword-based intent heuristics
# ---------------------------------------------------------------------------


def _looks_like_general_chat(normalized_message: str) -> bool:
    return any(marker in normalized_message for marker in ("你好", "您好", "hello", "hi", "天气", "翻译"))


def _looks_like_new_discovery(normalized_message: str) -> bool:
    return any(
        marker in normalized_message
        for marker in ("调研", "文献", "论文", "paper", "papers", "survey", "search", "find papers", "找相关文章")
    )


def _looks_like_scoped_paper_follow_up(normalized_message: str) -> bool:
    if not normalized_message:
        return False
    referential_markers = (
        "这篇",
        "该论文",
        "上一篇",
        "上一个",
        "这些论文",
        "这项工作",
        "本文",
        "this paper",
        "these papers",
        "current paper",
        "selected paper",
        "selected papers",
    )
    if any(marker in normalized_message for marker in referential_markers):
        return True
    paper_markers = ("论文", "paper", "papers", "文献", "work", "works")
    detail_markers = (
        "方法",
        "技术路线",
        "核心思路",
        "主要思路",
        "怎么做",
        "做法",
        "模型",
        "架构",
        "实验",
        "贡献",
        "创新点",
        "结果",
        "结论",
        "解释",
        "讲解",
        "method",
        "methods",
        "approach",
        "pipeline",
        "architecture",
        "experiment",
        "experiments",
        "contribution",
        "contributions",
        "results",
        "explain",
    )
    return any(marker in normalized_message for marker in paper_markers) and any(
        marker in normalized_message for marker in detail_markers
    )


def _looks_like_scoped_recommendation_request(normalized_message: str) -> bool:
    if not normalized_message:
        return False
    recommend_markers = ("推荐", "recommend", "suggest", "worth", "值得看", "值得读")
    paper_markers = ("论文", "paper", "papers", "work", "works", "文献")
    referential_markers = (
        "这篇",
        "这些",
        "当前",
        "候选",
        "已选",
        "勾选",
        "this paper",
        "these papers",
        "current papers",
        "selected papers",
        "candidate papers",
        "among",
    )
    curation_markers = (
        "精读",
        "先读",
        "先看",
        "优先读",
        "优先看",
        "必读",
        "代表性",
        "代表论文",
        "哪篇",
        "哪一个",
        "which one",
    )
    return (
        any(marker in normalized_message for marker in recommend_markers)
        and any(marker in normalized_message for marker in paper_markers)
        and (
            any(marker in normalized_message for marker in referential_markers)
            or any(marker in normalized_message for marker in curation_markers)
            or _looks_like_scoped_paper_follow_up(normalized_message)
        )
    )


def _looks_like_preference_recommendation_request(normalized_message: str) -> bool:
    recommend_markers = ("推荐", "recommend", "suggest", "worth", "值得看", "值得读")
    paper_markers = ("论文", "paper", "papers", "work", "works", "文献")
    return (
        any(marker in normalized_message for marker in recommend_markers)
        and any(marker in normalized_message for marker in paper_markers)
        and not _looks_like_scoped_recommendation_request(normalized_message)
    )


# ---------------------------------------------------------------------------
# Scope inheritance and route mode resolution
# ---------------------------------------------------------------------------


def _should_inherit_snapshot_scope(
    *,
    request: ResearchAgentRunRequest,
    snapshot,
) -> bool:
    if request.task_id or request.selected_paper_ids or request.selected_document_ids:
        return True
    if request.document_file_path or request.chart_image_path or request.document_id or request.chart_id:
        return False
    message = _normalize_topic_text_impl(request.message)
    if not message or _looks_like_general_chat(message):
        return False
    if _looks_like_new_discovery(message):
        return False
    if snapshot.active_route_mode == "general_chat":
        return False
    follow_up_markers = ("这篇", "该论文", "上一个", "这些论文", "this paper", "these papers", "p1", "p2")
    return any(marker in request.message.lower() for marker in follow_up_markers) or bool(snapshot.active_paper_ids)


def _route_mode_hint_for_request(
    *,
    request: ResearchAgentRunRequest,
    snapshot,
    inherit_scope: bool,
    intent_result: Any | None = None,
) -> str:
    if intent_result is not None:
        confidence = getattr(intent_result, "confidence", 0.0)
        intent_name = getattr(intent_result, "intent", "")
        if confidence >= 0.7 and intent_name in _INTENT_TO_ROUTE_MODE:
            return _INTENT_TO_ROUTE_MODE[intent_name]

    message = _normalize_topic_text_impl(request.message)
    if _looks_like_general_chat(message):
        return "general_chat"
    if request.chart_image_path or request.chart_id:
        return "chart_drilldown"
    if request.document_file_path or request.document_id:
        return "document_drilldown"
    has_scoped_papers = bool(
        request.selected_paper_ids
        or request.selected_document_ids
        or snapshot.active_paper_ids
        or snapshot.selected_paper_ids
    )
    if inherit_scope and has_scoped_papers and _looks_like_scoped_paper_follow_up(message):
        return "paper_follow_up"
    if _looks_like_new_discovery(message):
        return "research_discovery"
    if inherit_scope and has_scoped_papers:
        return "paper_follow_up"
    return snapshot.active_route_mode or "research_follow_up"


# ---------------------------------------------------------------------------
# Intent flags resolution (extracted from _state_from_context)
# ---------------------------------------------------------------------------


@dataclass
class IntentFlags:
    """Resolved intent signals computed from the user message and request metadata."""

    compare_requested: bool = False
    recommend_requested: bool = False
    preference_recommendation_requested: bool = False
    explain_requested: bool = False
    paper_detail_requested: bool = False
    paper_analysis_requested: bool = False
    analysis_focus: str | None = None
    context_compression_needed: bool = False


def resolve_intent_flags(
    *,
    research_goal_lower: str,
    advanced_action: str | None,
    comparison_dimensions: list[str],
    recommendation_goal: str | None,
    selected_paper_ids: list[str],
    active_paper_ids: list[str],
    paper_count: int,
    has_task: bool,
    has_papers: bool,
    session_history_count: int,
    context_compressed: bool,
    force_context_compression: bool,
    context_size_large: bool,
) -> IntentFlags:
    """Compute intent flags from the user message and request metadata.

    This is a pure function — it depends only on its parameters, not on any
    runtime instance state.
    """
    compare_requested = any(
        marker in research_goal_lower
        for marker in ("对比", "比较", "compare", "comparison", " vs ", "versus")
    ) or advanced_action == "compare" or bool(comparison_dimensions)

    recommend_requested = any(
        marker in research_goal_lower
        for marker in ("推荐", "recommend", "suggest", "建议阅读")
    ) or advanced_action == "recommend" or bool(recommendation_goal)

    preference_recommendation_requested = (
        advanced_action is None
        and _looks_like_preference_recommendation_request(research_goal_lower)
        and not compare_requested
        and not bool(recommendation_goal)
    )

    explain_requested = any(
        marker in research_goal_lower
        for marker in ("分析", "讲解", "解释", "怎么理解", "analysis", "analyze", "explain")
    ) or advanced_action == "analyze"

    paper_detail_requested = any(
        marker in research_goal_lower
        for marker in (
            "方法", "用了什么方法", "技术路线", "核心思路", "主要思路",
            "怎么做的", "做法", "模型", "架构", "实验", "贡献", "创新点",
            "method", "methods", "approach", "pipeline", "architecture",
            "experiment", "experiments", "contribution", "contributions",
        )
    )

    paper_analysis_requested = (
        compare_requested
        or (recommend_requested and not preference_recommendation_requested)
        or advanced_action == "analyze"
        or (
            bool(selected_paper_ids or active_paper_ids)
            and (explain_requested or paper_detail_requested)
        )
    )

    analysis_focus = (
        "compare" if compare_requested
        else "recommend" if recommend_requested
        else "explain" if explain_requested or paper_detail_requested
        else None
    )

    context_compression_needed = has_papers and not context_compressed and (
        force_context_compression
        or paper_analysis_requested
        or len(selected_paper_ids) >= 2
        or paper_count >= 4
        or session_history_count >= 6
        or context_size_large
    )

    return IntentFlags(
        compare_requested=compare_requested,
        recommend_requested=recommend_requested,
        preference_recommendation_requested=preference_recommendation_requested,
        explain_requested=explain_requested,
        paper_detail_requested=paper_detail_requested,
        paper_analysis_requested=paper_analysis_requested,
        analysis_focus=analysis_focus,
        context_compression_needed=context_compression_needed,
    )


# ---------------------------------------------------------------------------
# Force-finalize guardrail (extracted from ResearchRuntimeBase)
# ---------------------------------------------------------------------------

_TERMINAL_TASK_TYPES = {
    "general_answer",
    "answer_question",
    "analyze_papers",
    "analyze_paper_figures",
    "sync_to_zotero",
}


def should_force_finalize(
    *,
    exhausted: bool,
    stagnant_count: int,
    repeated_count: int,
    mode: str,
    has_qa_result: bool,
    latest_task_type: str | None,
    latest_status: str | None,
    latest_next_actions: set[str],
    workflow_constraint: str,
    has_preference_result: bool,
    advanced_action: str | None,
    has_paper_analysis: bool,
    new_topic_detected: bool,
    has_task_response: bool,
    has_report: bool,
    auto_import: bool,
    has_message: bool,
    import_attempted: bool,
    has_import_result: bool,
) -> bool:
    """Pure-function guardrail that decides if the supervisor loop should stop.

    Extracted from ``ResearchRuntimeBase._should_force_finalize`` so it can be
    tested and reasoned about independently of the runtime instance.
    """
    if exhausted:
        return True
    if stagnant_count >= 2 or repeated_count >= 2:
        return True
    if (
        latest_task_type == "answer_question"
        and (has_qa_result or latest_status == "failed")
    ):
        return True
    if mode == "qa" and has_qa_result:
        return True
    has_latest = latest_task_type is not None
    if (
        workflow_constraint == "discovery_only"
        and has_latest
        and latest_task_type == "search_literature"
        and latest_status == "succeeded"
        and has_report
    ):
        return True
    if has_preference_result:
        return True
    if advanced_action in {"analyze", "compare", "recommend"} and has_paper_analysis:
        return True
    if (
        not new_topic_detected
        and has_latest
        and latest_task_type == "search_literature"
        and latest_status == "succeeded"
        and has_task_response
        and has_report
        and mode != "qa"
        and not latest_next_actions.intersection({"write_review", "import_papers", "answer_question"})
    ):
        return True
    if (
        not new_topic_detected
        and has_latest
        and latest_task_type == "write_review"
        and latest_status == "succeeded"
        and has_task_response
        and has_report
        and mode != "qa"
        and not latest_next_actions.intersection({"import_papers", "answer_question"})
    ):
        return True
    if (
        workflow_constraint != "discovery_only"
        and not new_topic_detected
        and has_task_response
        and has_report
        and not auto_import
        and mode != "qa"
    ):
        return True
    if import_attempted and has_import_result and not has_message:
        return True
    # Fast-finalize: terminal single-shot actions
    if (
        has_latest
        and latest_status in {"succeeded", "skipped"}
        and latest_task_type in _TERMINAL_TASK_TYPES
        and not latest_next_actions
    ):
        return True
    return False
