"""Compress context supervisor tool."""

from __future__ import annotations

from agents.research_supervisor_agent import ResearchSupervisorDecision
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.supervisor_tools.mixins import _WorkspacePersistenceMixin
from services.research.unified_action_adapters import (
    build_context_compression_input,
    build_context_compression_output,
)


class CompressContextTool(_WorkspacePersistenceMixin):
    name = "compress_context"

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        execution_context = context.execution_context
        if execution_context is None or execution_context.research_context is None:
            return ResearchToolResult(status="skipped", observation="no execution context is available for compression", metadata={"reason": "missing_execution_context"})
        compression_input = build_context_compression_input(context=context, decision=decision)
        selected_paper_ids = compression_input.resolved_selected_paper_ids()
        compressed = context.research_service.research_context_manager.compress_papers(
            papers=list(context.papers),
            selected_paper_ids=selected_paper_ids,
            paper_reading_skill=context.research_service.paper_reading_skill,
        )
        if not compressed:
            return ResearchToolResult(status="skipped", observation="no paper summary could be built for compression", metadata={"reason": "no_papers"})
        updated_context = context.research_service.research_context_manager.update_context(
            current_context=execution_context.research_context,
            selected_papers=selected_paper_ids,
            paper_summaries=compressed,
            metadata={
                "context_compression": {
                    "paper_count": len({summary.paper_id for summary in compressed}),
                    "summary_count": len(compressed),
                    "levels": sorted({summary.level for summary in compressed}),
                }
            },
        )
        execution_context.research_context = updated_context
        execution_context.context_slices = context.research_service.build_context_slices(
            updated_context,
            selected_paper_ids=selected_paper_ids,
        )
        context.compressed_context_summary = {
            "paper_count": len({summary.paper_id for summary in compressed}),
            "summary_count": len(compressed),
            "levels": sorted({summary.level for summary in compressed}),
            "compressed_paper_ids": list(dict.fromkeys(summary.paper_id for summary in compressed)),
        }
        self._persist_workspace_results(context, compression_summary=context.compressed_context_summary)
        output = build_context_compression_output(
            compression_summary=context.compressed_context_summary,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"context compressed; papers={context.compressed_context_summary['paper_count']}; summaries={context.compressed_context_summary['summary_count']}"
            ),
            metadata=output.to_metadata(),
        )
