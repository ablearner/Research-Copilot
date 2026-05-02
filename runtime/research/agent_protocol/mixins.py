"""Shared standalone utility functions for workspace persistence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.schemas.research import PaperCandidate, ResearchReport, ResearchTask, ResearchTaskResponse
from domain.schemas.research_functions import AnalyzePapersFunctionOutput
from core.utils import now_iso as _now_iso
from runtime.research.agent_protocol.base import ResearchAgentToolContext


@dataclass
class WorkspaceUpdateResult:
    """Pure data produced by persist_workspace_results when persist=False."""
    updated_task: ResearchTask | None = None
    updated_report: ResearchReport | None = None
    updated_task_response: ResearchTaskResponse | None = None
    save_event_type: str = "memory_updated"
    save_event_payload: dict[str, Any] = field(default_factory=dict)
    memory_save_context_params: dict[str, Any] | None = None


def _dedupe_ids(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _dedupe_text(values: list[str], *, limit: int) -> list[str]:
    deduped = [value.strip() for value in values if value and value.strip()]
    return list(dict.fromkeys(deduped))[:limit]


def comparison_scope_papers(
    *,
    papers: list[PaperCandidate],
    selected_paper_ids: list[str],
) -> list[PaperCandidate]:
    if selected_paper_ids:
        allowed = set(selected_paper_ids)
        resolved = [paper for paper in papers if paper.paper_id in allowed]
        if resolved:
            return resolved
    ranked = list(papers)
    ranked.sort(
        key=lambda paper: (
            float(paper.relevance_score or 0.0),
            int(paper.citations or 0),
            int(paper.year or 0),
        ),
        reverse=True,
    )
    return ranked[:3]


def persist_workspace_results(
    context: ResearchAgentToolContext,
    *,
    paper_analysis: AnalyzePapersFunctionOutput | None = None,
    analyzed_papers: list[PaperCandidate] | None = None,
    compression_summary: dict[str, Any] | None = None,
    persist: bool = True,
) -> WorkspaceUpdateResult | None:
    task_response = context.task_response
    execution_context = context.execution_context
    if task_response is None:
        return
    task = task_response.task
    workspace_metadata = dict(task.workspace.metadata)
    key_findings = list(task.workspace.key_findings)
    next_actions = list(task.workspace.next_actions)
    must_read_ids = list(task.workspace.must_read_paper_ids)
    selected_paper_ids: list[str] = list(context.request.selected_paper_ids)
    if paper_analysis is not None:
        workspace_metadata["latest_paper_analysis"] = paper_analysis.model_dump(mode="json")
        key_findings.append(paper_analysis.answer)
        analyzed_ids = [paper.paper_id for paper in analyzed_papers or []]
        selected_paper_ids.extend(analyzed_ids)
        recommended_ids = list(paper_analysis.recommended_paper_ids)
        must_read_ids = _dedupe_ids([*must_read_ids, *recommended_ids])
        if paper_analysis.focus == "compare":
            next_actions.append("可以继续围绕这组论文追问更细的实验差异、适用场景或失败案例。")
        elif paper_analysis.focus == "recommend":
            next_actions.append("可以直接导入推荐论文全文，或围绕推荐理由继续提问。")
        else:
            next_actions.append("可以继续针对这组论文追问方法细节、实验设置或适用边界。")
    if compression_summary is not None:
        workspace_metadata["context_compression"] = dict(compression_summary)
    updated_at = _now_iso()
    updated_workspace = task.workspace.model_copy(
        update={
            "key_findings": _dedupe_text(key_findings, limit=6),
            "must_read_paper_ids": must_read_ids,
            "next_actions": _dedupe_text(next_actions, limit=5),
            "metadata": workspace_metadata,
        }
    )
    updated_task = task.model_copy(update={"updated_at": updated_at, "workspace": updated_workspace})
    updated_report = (
        task_response.report.model_copy(update={"workspace": updated_workspace})
        if task_response.report is not None
        else None
    )
    event_payload = {
        "tool_name": "workspace_persist",
        "context_compression": compression_summary,
        "has_paper_analysis": paper_analysis is not None,
    }
    updated_task_response = task_response.model_copy(update={"task": updated_task, "report": updated_report})

    memory_save_context_params = None
    if execution_context is not None and execution_context.research_context is not None:
        research_context = context.research_service.research_context_manager.update_context(
            current_context=execution_context.research_context,
            selected_papers=_dedupe_ids(selected_paper_ids),
            known_conclusions=_dedupe_text(key_findings, limit=6),
            metadata={
                **(
                    {"latest_paper_analysis": paper_analysis.model_dump(mode="json")}
                    if paper_analysis is not None
                    else {}
                ),
                **({"context_compression": compression_summary} if compression_summary is not None else {}),
            },
        )
        if selected_paper_ids:
            research_context.active_papers = _dedupe_ids(selected_paper_ids)
        execution_context.research_context = research_context
        execution_context.context_slices = context.research_service.build_context_slices(
            research_context,
            selected_paper_ids=_dedupe_ids(selected_paper_ids),
        )
        if execution_context.session_id:
            memory_save_context_params = {
                "session_id": execution_context.session_id,
                "research_context": research_context,
            }

    if persist:
        context.research_service.save_task_state(
            updated_task,
            conversation_id=context.request.conversation_id,
            event_type="memory_updated",
            payload=event_payload,
        )
        if updated_report is not None:
            context.research_service.report_service.save_report(updated_report)
        context.task_response = updated_task_response
        if memory_save_context_params is not None:
            context.research_service.memory_gateway.save_context(
                memory_save_context_params["session_id"],
                memory_save_context_params["research_context"],
            )
        return None

    return WorkspaceUpdateResult(
        updated_task=updated_task,
        updated_report=updated_report,
        updated_task_response=updated_task_response,
        save_event_type="memory_updated",
        save_event_payload=event_payload,
        memory_save_context_params=memory_save_context_params,
    )
