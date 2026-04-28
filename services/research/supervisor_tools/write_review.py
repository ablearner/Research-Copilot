"""Write review supervisor tool."""

from __future__ import annotations

from typing import Any

from agents.research_supervisor_agent import ResearchSupervisorDecision
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.research import ResearchReport
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from core.utils import now_iso as _now_iso
from services.research.supervisor_tools.mixins import _PlannerMessageTool
from services.research.unified_action_adapters import (
    build_review_draft_input,
    build_review_draft_output,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.research.literature_research_service import LiteratureResearchService


class WriteReviewTool(_PlannerMessageTool):
    name = "write_review"

    def __init__(self, *, research_service: LiteratureResearchService, writer_agent: ResearchWriterAgent) -> None:
        self.research_service = research_service
        self.writer_agent = writer_agent

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for review drafting",
                metadata={"reason": "missing_task"},
            )

        review_input = build_review_draft_input(context=context)
        existing_report = review_input.report
        candidate_report = existing_report or self.writer_agent.synthesize(
            review_input
        )
        retry_count = 0
        quality = self._quality_metrics(candidate_report)
        if not quality["passed"]:
            retry_count += 1
            candidate_report = self.writer_agent.synthesize(review_input)
            quality = self._quality_metrics(candidate_report)

        generated_at = _now_iso()
        task = task_response.task
        saved_report = candidate_report.model_copy(
            update={
                "generated_at": generated_at,
                "metadata": {
                    **candidate_report.metadata,
                    "worker_agent": "ResearchWriterAgent",
                    "write_review_retry_count": retry_count,
                    "write_review_quality_passed": quality["passed"],
                    "write_review_issue_count": len(quality["issues"]),
                },
            }
        )
        updated_task = task.model_copy(
            update={
                "updated_at": generated_at,
                "report_id": saved_report.report_id,
                "workspace": task.workspace.model_copy(
                    update={
                        "metadata": {
                            **task.workspace.metadata,
                            "write_review_retry_count": retry_count,
                        }
                    }
                ),
            }
        )
        request = context.request
        self.research_service.report_service.save_report(saved_report)
        self.research_service.save_task_state(
            updated_task,
            conversation_id=request.conversation_id,
            event_type="memory_updated",
            payload={"tool_name": "write_review", "report_id": saved_report.report_id},
        )
        context.task_response = task_response.model_copy(update={"task": updated_task, "report": saved_report})
        context.execution_context = self.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=updated_task,
            report=saved_report,
            papers=review_input.curated_papers,
            document_ids=updated_task.imported_document_ids,
            selected_paper_ids=request.selected_paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=request.metadata,
        )
        output = build_review_draft_output(
            task_id=updated_task.task_id,
            report_id=saved_report.report_id,
            quality=quality,
            retry_count=retry_count,
        )
        return ResearchToolResult(
            status="succeeded" if quality["passed"] else "failed",
            observation=(
                f"review drafted; words={quality['word_count']}; citations={quality['has_citations']}; retries={retry_count}"
            ),
            metadata=output.to_metadata(),
        )

    def _quality_metrics(self, report: ResearchReport) -> dict[str, Any]:
        text = report.markdown
        word_count = len([token for token in text.replace("\n", " ").split(" ") if token.strip()])
        has_citations = "[P" in text or "引用" in text
        has_key_sections = all(
            section in text
            for section in ("## 研究背景", "## 核心问题", "## 方法对比", "## 关键发现")
        ) or all(
            section in text
            for section in ("## 研究背景", "## 方法对比", "## 关键发现")
        )
        issues: list[str] = []
        if word_count < 250:
            issues.append("review_too_short")
        if not has_citations:
            issues.append("missing_citations")
        if not has_key_sections:
            issues.append("missing_required_sections")
        return {
            "passed": not issues,
            "word_count": word_count,
            "has_citations": has_citations,
            "has_key_sections": has_key_sections,
            "issues": issues,
        }
