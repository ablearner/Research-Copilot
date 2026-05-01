"""Answer research question supervisor tools."""

from __future__ import annotations

from agents.research_supervisor_agent import ResearchSupervisorDecision
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.unified_action_adapters import (
    build_collection_qa_input,
    build_collection_qa_output,
    resolve_active_message,
)


class AnswerResearchQuestionTool:
    name = "answer_research_question"

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        request = context.request
        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for collection QA",
                metadata={"reason": "missing_task"},
            )

        active_message = resolve_active_message(decision)
        qa_input = build_collection_qa_input(
            context=context,
            task_id=task_response.task.task_id,
            active_message=active_message,
        )
        qa_result = await context.research_service.ask_task_collection(
            task_response.task.task_id,
            qa_input.to_research_task_ask_request(),
            graph_runtime=context.graph_runtime,
        )
        context.qa_result = qa_result
        context.task_response = context.research_service.get_task(task_response.task.task_id)
        context.execution_context = context.research_service.build_execution_context(
            graph_runtime=context.graph_runtime,
            conversation_id=request.conversation_id,
            task=context.task_response.task if context.task_response else None,
            report=context.task_response.report if context.task_response else None,
            papers=context.task_response.papers if context.task_response else None,
            document_ids=list(qa_result.document_ids),
            selected_paper_ids=list(qa_result.paper_ids),
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=qa_result.qa.metadata if isinstance(qa_result.qa.metadata, dict) else request.metadata,
        )
        if request.conversation_id:
            context.research_service.record_qa_turn(
                request.conversation_id,
                task_response=context.task_response,
                ask_response=qa_result,
            )
        if context.execution_context is not None and context.execution_context.session_id and qa_result.paper_ids:
            context.research_service.memory_gateway.set_active_papers(
                context.execution_context.session_id,
                list(qa_result.paper_ids),
            )
        output = build_collection_qa_output(qa_result=qa_result)
        metadata = output.to_metadata()
        qa_metadata = qa_result.qa.metadata if isinstance(qa_result.qa.metadata, dict) else {}
        quality_check = qa_metadata.get("answer_quality_check")
        if isinstance(quality_check, dict) and quality_check.get("needs_recovery"):
            recovery_route = str(quality_check.get("suggested_recovery_qa_route") or "").strip()
            if recovery_route:
                metadata.update(
                    {
                        "reason": "qa_route_replan_requested",
                        "observation_envelope": {
                            "progress_made": False,
                            "confidence": qa_result.qa.confidence,
                            "suggested_next_actions": ["answer_question"],
                            "state_delta": {
                                "preferred_qa_route": recovery_route,
                                "qa_recovery_reason": quality_check.get("suggested_recovery_rationale"),
                            },
                        },
                    }
                )
                return ResearchToolResult(
                    status="skipped",
                    observation=(
                        "research QA produced an under-supported answer; supervisor replan requested "
                        f"with qa_route={recovery_route}"
                    ),
                    metadata=metadata,
                )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"answered research collection question; evidence={len(qa_result.qa.evidence_bundle.evidences)}; "
                f"confidence={qa_result.qa.confidence if qa_result.qa.confidence is not None else 'empty'}"
            ),
            metadata=metadata,
        )


class AnswerQuestionTool(AnswerResearchQuestionTool):
    name = "answer_question"
