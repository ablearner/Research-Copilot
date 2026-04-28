"""Shared mixin classes used by multiple supervisor tools."""

from __future__ import annotations

from typing import Any

from domain.schemas.agent_message import AgentMessage
from domain.schemas.research import PaperCandidate, ResearchReport
from domain.schemas.research_functions import AnalyzePapersFunctionOutput
from agents.research_supervisor_agent import ResearchSupervisorDecision
from core.utils import now_iso as _now_iso
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)


class _PlannerMessageTool:
    def _active_message(self, decision: ResearchSupervisorDecision) -> AgentMessage | None:
        if not isinstance(decision.metadata, dict):
            return None
        active_message = decision.metadata.get("active_message")
        if isinstance(active_message, AgentMessage):
            return active_message
        if isinstance(active_message, dict):
            return AgentMessage.model_validate(active_message)
        return None


class _WorkspacePersistenceMixin:
    def _dedupe_ids(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def _dedupe_text(self, values: list[str], *, limit: int) -> list[str]:
        deduped = [value.strip() for value in values if value and value.strip()]
        return list(dict.fromkeys(deduped))[:limit]

    def _comparison_scope_papers(
        self,
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

    def _persist_workspace_results(
        self,
        context: ResearchAgentToolContext,
        *,
        paper_analysis: AnalyzePapersFunctionOutput | None = None,
        analyzed_papers: list[PaperCandidate] | None = None,
        compression_summary: dict[str, Any] | None = None,
    ) -> None:
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
            must_read_ids = self._dedupe_ids([*must_read_ids, *recommended_ids])
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
                "key_findings": self._dedupe_text(key_findings, limit=6),
                "must_read_paper_ids": must_read_ids,
                "next_actions": self._dedupe_text(next_actions, limit=5),
                "metadata": workspace_metadata,
            }
        )
        updated_task = task.model_copy(update={"updated_at": updated_at, "workspace": updated_workspace})
        updated_report = (
            task_response.report.model_copy(update={"workspace": updated_workspace})
            if task_response.report is not None
            else None
        )
        context.research_service.save_task_state(
            updated_task,
            conversation_id=context.request.conversation_id,
            event_type="memory_updated",
            payload={
                "tool_name": "workspace_persist",
                "context_compression": compression_summary,
                "has_paper_analysis": paper_analysis is not None,
            },
        )
        if updated_report is not None:
            context.research_service.report_service.save_report(updated_report)
        context.task_response = task_response.model_copy(update={"task": updated_task, "report": updated_report})
        if execution_context is not None and execution_context.research_context is not None:
            research_context = context.research_service.research_context_manager.update_context(
                current_context=execution_context.research_context,
                selected_papers=self._dedupe_ids(selected_paper_ids),
                known_conclusions=self._dedupe_text(key_findings, limit=6),
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
                research_context.active_papers = self._dedupe_ids(selected_paper_ids)
            execution_context.research_context = research_context
            execution_context.context_slices = context.research_service.build_context_slices(
                research_context,
                selected_paper_ids=self._dedupe_ids(selected_paper_ids),
            )
            if execution_context.session_id:
                context.research_service.memory_manager.save_context(
                    execution_context.session_id,
                    research_context,
                )
