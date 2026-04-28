"""Understand chart supervisor tool."""

from __future__ import annotations

from agents.chart_analysis_agent import ChartAnalysisAgent
from agents.research_supervisor_agent import ResearchSupervisorDecision
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.unified_action_adapters import (
    build_chart_understanding_input,
    build_chart_understanding_output,
)


class UnderstandChartTool:
    name = "understand_chart"

    def __init__(self, *, chart_analysis_agent: ChartAnalysisAgent) -> None:
        self.chart_analysis_agent = chart_analysis_agent

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        context.chart_attempted = True
        chart_input = build_chart_understanding_input(context=context, decision=decision)
        if not chart_input.image_path:
            return ResearchToolResult(
                status="skipped",
                observation="no chart_image_path was provided for chart understanding",
                metadata={"reason": "missing_chart_image_path"},
            )

        chart_result = await self.chart_analysis_agent.understand_chart(
            graph_runtime=context.graph_runtime,
            image_path=chart_input.image_path,
            document_id=chart_input.document_id,
            page_id=chart_input.page_id,
            page_number=chart_input.page_number,
            chart_id=chart_input.chart_id,
            session_id=chart_input.session_id,
            context=chart_input.context,
            skill_name=chart_input.skill_name,
        )
        context.chart_result = chart_result
        chart = getattr(chart_result, "chart", None)
        output = build_chart_understanding_output(
            chart_result=chart_result,
            chart_input=chart_input,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"chart understood; chart_id={getattr(chart, 'id', chart_input.chart_id)}; "
                f"chart_type={getattr(chart, 'chart_type', 'unknown')}"
            ),
            metadata=output.to_metadata(),
        )
