from __future__ import annotations

from typing import Any

from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research import (
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchAgentTraceStep,
)
from runtime.research.agent_protocol import ResearchAgentToolContext
from runtime.research.unified_runtime import (
    serialize_unified_agent_messages,
    serialize_unified_agent_registry,
    serialize_unified_agent_results,
    serialize_unified_delegation_plan,
)


class ResearchAgentResultAggregator:
    """Build the final research-agent response from state and tool context."""

    def __init__(self, *, runtime: Any) -> None:
        self.runtime = runtime

    def build_response(
        self,
        *,
        request: ResearchAgentRunRequest,
        context: ResearchAgentToolContext,
        trace: list[ResearchAgentTraceStep],
        failed: bool,
        exhausted: bool,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        clarification_request: str | None,
        active_plan_id: str | None,
    ) -> ResearchAgentRunResponse:
        warnings = list(context.warnings or [])
        if context.task_response:
            warnings.extend(warning for warning in context.task_response.warnings if warning not in warnings)
        if context.import_result:
            for result in context.import_result.results:
                if result.status == "failed" and result.error_message:
                    warning = f"import failed: {result.title}: {result.error_message}"
                    if warning not in warnings:
                        warnings.append(warning)
        if clarification_request and clarification_request not in warnings:
            warnings.append(clarification_request)
        workspace = self.runtime._resolve_workspace(
            context=context,
            trace=trace,
            failed=failed,
            exhausted=exhausted,
        )
        advanced_strategy = self.runtime._resolved_advanced_strategy(
            context,
            workspace=workspace,
        )
        workspace = workspace.model_copy(
            update={
                "metadata": {
                    **workspace.metadata,
                    "advanced_strategy": advanced_strategy.model_dump(mode="json"),
                }
            }
        )
        context.task_response = self.runtime.research_service.persist_runtime_state(
            task_response=context.task_response,
            workspace=workspace,
            conversation_id=request.conversation_id,
            advanced_strategy=advanced_strategy,
        )
        messages = self.runtime._build_messages(
            request,
            context,
            trace,
            workspace,
            agent_messages=agent_messages,
            agent_results=agent_results,
            clarification_request=clarification_request,
            replan_count=replan_count,
        )
        self.runtime._log_internal_runtime_state(
            request=request,
            workspace=workspace,
            trace=trace,
            agent_messages=agent_messages,
            agent_results=agent_results,
            clarification_request=clarification_request,
        )
        next_actions = self.runtime._next_actions(
            context,
            workspace,
            clarification_request=clarification_request,
        )
        status = "failed" if failed and not context.task_response else "partial" if failed or warnings else "succeeded"
        runtime_metadata = self.runtime._response_runtime_metadata()
        active_paper_ids = (
            list(context.execution_context.research_context.active_papers)
            if context.execution_context and context.execution_context.research_context
            else self.runtime._active_paper_ids_for_manager(context)
        )
        capability_summary = (
            self.runtime.capability_registry.inventory_summary(graph_runtime=context.graph_runtime)
            if getattr(self.runtime, "capability_registry", None) is not None
            else {
                "local_capability_count": len(self.runtime.tool_registry.filter_by_toolset("supervisor_action")),
                "action_count": len(self.runtime.tool_registry.filter_by_toolset("supervisor_action")),
                "knowledge_count": 0,
                "runtime_count": 0,
                "mcp_server_count": 0,
                "action_names": [],
                "knowledge_names": [],
                "runtime_names": [],
                "mcp_server_names": [],
            }
        )
        skill_metadata = (
            context.skill_selection.metadata()
            if context.skill_selection is not None
            else {}
        )
        action_tool_trace_count = len(self.runtime.action_tool_executor.get_traces())
        specialist_execution_trace_count = len(agent_results)
        return ResearchAgentRunResponse(
            status=status,
            task=context.task,
            papers=context.papers,
            report=context.report,
            import_result=context.import_result,
            qa=context.qa_result.qa if context.qa_result else None,
            parsed_document=context.parsed_document,
            document_index_result=context.document_index_result,
            chart=getattr(context.chart_result, "chart", None) if context.chart_result is not None else None,
            chart_graph_text=getattr(context.chart_result, "graph_text", None) if context.chart_result is not None else None,
            messages=messages,
            trace=trace,
            warnings=warnings,
            next_actions=next_actions,
            workspace=workspace,
            metadata={
                **runtime_metadata,
                "tool_count": capability_summary["local_capability_count"],
                "capability_inventory": capability_summary,
                "supervisor_action_tool_engine": "tool_executor",
                "supervisor_worker_execution_engine": (
                    "unified_agent_registry" if context.unified_agent_registry is not None else "action_tool_fallback"
                ),
                "supervisor_action_trace_count": action_tool_trace_count or specialist_execution_trace_count,
                "specialist_execution_trace_count": specialist_execution_trace_count,
                "unified_supervisor_mode": "pure_supervisor",
                "unified_runtime_blueprint": context.unified_blueprint or {},
                "unified_agent_registry": serialize_unified_agent_registry(context.unified_agent_registry),
                "task_id": context.task.task_id if context.task else None,
                "qa_task_id": context.qa_result.task_id if context.qa_result else None,
                "active_paper_ids": active_paper_ids,
                "route_mode": (
                    ((context.request.metadata or {}).get("context") or {}).get("route_mode")
                    if isinstance((context.request.metadata or {}).get("context"), dict)
                    else None
                ),
                "active_thread_id": (
                    ((context.request.metadata or {}).get("context") or {}).get("active_thread_id")
                    if isinstance((context.request.metadata or {}).get("context"), dict)
                    else None
                ),
                "session_id": context.execution_context.session_id if context.execution_context else None,
                "memory_enabled": context.execution_context.memory_enabled if context.execution_context else False,
                "has_document_tool_output": context.parsed_document is not None,
                "has_chart_tool_output": context.chart_result is not None,
                "has_general_answer": context.general_answer is not None,
                "preference_recommendations": (
                    context.preference_recommendation_result.model_dump(mode="json")
                    if context.preference_recommendation_result is not None
                    else workspace.metadata.get("latest_preference_recommendations")
                ),
                "workspace_stage": workspace.current_stage,
                "workspace_summary": workspace.status_summary,
                "stop_reason": workspace.stop_reason,
                "trace_steps": len(trace),
                "manager_decision_count": planner_runs,
                "recovery_decision_count": replan_count,
                "agent_message_count": len(agent_messages),
                "agent_result_count": len(agent_results),
                "active_decision_batch_id": active_plan_id,
                "clarification_requested": bool(clarification_request),
                "advanced_strategy": advanced_strategy.model_dump(mode="json"),
                "delegation_plan": self.runtime._serialize_task_plan(agent_messages, agent_results),
                "unified_delegation_plan": serialize_unified_delegation_plan(
                    agent_messages,
                    agent_results,
                    registry=context.unified_agent_registry,
                ),
                "agent_lane_states": (
                    {
                        name: state.model_dump(mode="json")
                        for name, state in context.execution_context.research_context.sub_manager_states.items()
                    }
                    if context.execution_context and context.execution_context.research_context
                    else {}
                ),
                "agent_messages": [message.model_dump(mode="json") for message in agent_messages],
                "agent_results": [result.model_dump(mode="json") for result in agent_results],
                "general_answer": context.general_answer,
                "general_answer_metadata": context.general_answer_metadata or {},
                "active_skills": list(skill_metadata.get("active_skill_names") or []),
                "skill_selection": skill_metadata,
                "unified_agent_messages": serialize_unified_agent_messages(
                    agent_messages,
                    registry=context.unified_agent_registry,
                ),
                "unified_agent_results": serialize_unified_agent_results(
                    agent_results,
                    registry=context.unified_agent_registry,
                ),
            },
        )
