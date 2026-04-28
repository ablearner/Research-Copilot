"""Import papers supervisor tools."""

from __future__ import annotations

from agents.research_supervisor_agent import ResearchSupervisorDecision
from domain.schemas.research import ImportPapersRequest
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.unified_action_adapters import (
    build_paper_import_input,
    build_paper_import_output,
)


class ImportRelevantPapersTool:
    name = "import_relevant_papers"

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        task_response = context.task_response
        context.import_attempted = True
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for paper import",
                metadata={"reason": "missing_task"},
            )

        import_input = build_paper_import_input(context=context, decision=decision)
        paper_ids = import_input.resolved_paper_ids(task_response.papers)
        if not paper_ids:
            return ResearchToolResult(
                status="skipped",
                observation="no importable paper with an available PDF was found",
                metadata={"reason": "no_import_candidates"},
            )

        import_result = await context.research_service.import_papers(
            ImportPapersRequest(
                task_id=task_response.task.task_id,
                paper_ids=paper_ids,
                include_graph=import_input.include_graph,
                include_embeddings=import_input.include_embeddings,
                skill_name=import_input.skill_name,
                conversation_id=import_input.conversation_id,
            ),
            graph_runtime=context.graph_runtime,
        )
        context.import_result = import_result
        refreshed = context.research_service.get_task(task_response.task.task_id)
        context.task_response = refreshed
        request = context.request
        context.execution_context = context.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=refreshed.task,
            report=refreshed.report,
            papers=refreshed.papers,
            document_ids=refreshed.task.imported_document_ids,
            selected_paper_ids=paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=request.metadata,
        )
        if request.conversation_id:
            context.research_service.record_import_turn(
                request.conversation_id,
                task_response=refreshed,
                import_response=import_result,
                selected_paper_ids=paper_ids,
            )
        output = build_paper_import_output(
            paper_ids=paper_ids,
            import_result=import_result,
        )
        return ResearchToolResult(
            status="succeeded" if import_result.failed_count == 0 else "failed" if import_result.imported_count == 0 and import_result.skipped_count == 0 else "succeeded",
            observation=(
                f"paper import finished; imported={import_result.imported_count}; "
                f"skipped={import_result.skipped_count}; failed={import_result.failed_count}"
            ),
            metadata=output.to_metadata(),
        )


class ImportPapersTool(ImportRelevantPapersTool):
    name = "import_papers"
