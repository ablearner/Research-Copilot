"""General answer supervisor tool."""

from __future__ import annotations

from agents.general_answer_agent import GeneralAnswerAgent
from agents.research_supervisor_agent import ResearchSupervisorDecision
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.unified_action_adapters import resolve_active_message


class GeneralAnswerTool:
    name = "general_answer"

    def __init__(self, *, general_answer_agent: GeneralAnswerAgent) -> None:
        self.general_answer_agent = general_answer_agent

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        question = str(payload.get("goal") or context.request.message or "").strip()
        on_token = None
        if context.progress_callback is not None:
            async def on_token(text: str) -> None:
                await context.progress_callback({"type": "token", "text": text})
        result = await self.general_answer_agent.answer(
            question=question,
            conversation_context={
                "mode": context.request.mode,
                "task_id": context.request.task_id,
                "has_task": context.task is not None,
                "selected_paper_ids": [] if payload.get("ignore_research_context") else list(context.request.selected_paper_ids),
                "ignore_research_context": bool(payload.get("ignore_research_context")),
            },
            on_token=on_token,
        )
        warnings = list(result.warnings)
        should_reroute = "route_mismatch" in warnings or (
            result.answer_type == "reroute_hint"
        ) or (
            result.confidence < 0.45 and (
                context.request.task_id is not None
                or bool(context.request.selected_paper_ids)
                or bool(context.request.selected_document_ids)
                or bool(context.request.chart_image_path)
                or bool(context.request.document_file_path)
            )
        )
        if should_reroute:
            return ResearchToolResult(
                status="skipped",
                observation="general answer agent detected a likely route mismatch and requested supervisor rerouting",
                metadata={
                    **result.model_dump(mode="json"),
                    "reason": "route_mismatch",
                    "suggested_action": self._suggested_action(context=context),
                },
            )
        context.general_answer = result.answer
        context.general_answer_metadata = result.model_dump(mode="json")
        return ResearchToolResult(
            status="succeeded",
            observation="general question answered directly without research workspace tools",
            metadata=context.general_answer_metadata,
        )

    def _suggested_action(self, *, context: ResearchAgentToolContext) -> str:
        if context.request.chart_image_path:
            return "understand_chart"
        if context.request.document_file_path:
            return "understand_document"
        if context.request.task_id or context.request.selected_paper_ids or context.request.selected_document_ids:
            return "answer_question"
        return "search_literature"
