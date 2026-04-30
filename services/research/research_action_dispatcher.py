from __future__ import annotations

from typing import Any
from uuid import uuid4

from domain.schemas.unified_runtime import (
    UNIFIED_ACTION_OUTPUT_METADATA_KEY,
    UnifiedAgentResult,
    UnifiedAgentTask,
)
from services.research.supervisor_tools import (
    ResearchAgentToolContext,
    ResearchToolResult,
    _observation_envelope,
)
from tooling.research_supervisor_tool_specs import SupervisorActionToolOutput


class ResearchActionDispatcher:
    """Single execution path for supervisor actions and unified worker fallback."""

    def __init__(self, *, runtime: Any) -> None:
        self.runtime = runtime

    def build_action_tool_handlers(self) -> dict[str, Any]:
        return {
            name: self.build_action_tool_handler(name)
            for name in self.runtime.tools
        }

    def build_action_tool_handler(self, action_name: str):
        async def handler(*, invocation_id: str):
            invocation = self.runtime._action_invocations.get(invocation_id)
            if invocation is None:
                raise RuntimeError(f"unknown supervisor action invocation: {invocation_id}")
            context, decision = invocation
            tool = self.runtime.tools.get(action_name)
            if tool is None:
                raise RuntimeError(f"supervisor action tool not registered: {action_name}")
            result = await tool.run(context, decision)
            return SupervisorActionToolOutput(
                status=result.status if result.status in {"succeeded", "failed", "skipped"} else "failed",
                observation=result.observation,
                metadata=dict(result.metadata),
            )

        return handler

    async def execute_action_tool(
        self,
        *,
        action_name: str,
        context: ResearchAgentToolContext,
        decision: Any,
    ):
        invocation_id = f"supervisor_action_{uuid4().hex}"
        self.runtime._action_invocations[invocation_id] = (context, decision)
        try:
            execution_result = await self.runtime.action_tool_executor.execute_tool_call(
                action_name,
                {"invocation_id": invocation_id},
            )
            normalized = self.normalize_execution_result_metadata(
                action_name=action_name,
                context=context,
                execution_result=execution_result,
            )
            if normalized is not None:
                execution_result.output = normalized
            return execution_result
        finally:
            self.runtime._action_invocations.pop(invocation_id, None)

    def normalize_execution_result_metadata(
        self,
        *,
        action_name: str,
        context: ResearchAgentToolContext,
        execution_result,
    ) -> SupervisorActionToolOutput | None:
        output = execution_result.output
        if output is None:
            return None
        if isinstance(output, SupervisorActionToolOutput):
            metadata = dict(output.metadata)
            metadata = self.with_standardized_observation(
                action_name=action_name,
                context=context,
                status=output.status,
                metadata=metadata,
            )
            return SupervisorActionToolOutput(
                status=output.status,
                observation=output.observation,
                metadata=metadata,
            )
        if isinstance(output, dict):
            metadata = self.with_standardized_observation(
                action_name=action_name,
                context=context,
                status=str(output.get("status") or "failed"),
                metadata=dict(output.get("metadata") or {}),
            )
            return SupervisorActionToolOutput(
                status=str(output.get("status") or "failed"),
                observation=str(output.get("observation") or execution_result.error_message or ""),
                metadata=metadata,
            )
        return None

    def with_standardized_observation(
        self,
        *,
        action_name: str,
        context: ResearchAgentToolContext,
        status: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        next_actions = list((context.workspace.next_actions if context.workspace is not None else [])[:3])
        observation = dict(metadata.get("observation_envelope") or {})
        progress_made = status == "succeeded" and metadata.get("reason") not in {"route_mismatch"}
        confidence = metadata.get("confidence")
        if action_name == "answer_question" and context.qa_result is not None:
            confidence = context.qa_result.qa.confidence
            next_actions = list((context.qa_result.report.workspace.next_actions if context.qa_result.report is not None else next_actions)[:3])
        elif action_name == "general_answer":
            confidence = metadata.get("confidence")
        elif action_name == "search_literature":
            confidence = 0.82 if status == "succeeded" else 0.35
        elif action_name == "write_review":
            confidence = 0.78 if status == "succeeded" else 0.4
        elif action_name == "analyze_papers":
            confidence = 0.8 if status == "succeeded" else 0.4
        elif action_name == "recommend_from_preferences":
            confidence = 0.83 if status == "succeeded" else 0.35
        elif action_name == "compress_context":
            confidence = 0.74 if status == "succeeded" else 0.3
        missing_inputs = list(observation.get("missing_inputs") or [])
        reason = str(metadata.get("reason") or "").strip()
        if reason == "missing_task":
            missing_inputs.append("task")
        elif reason == "missing_document_file_path":
            missing_inputs.append("document_file_path")
        elif reason == "missing_chart_image_path":
            missing_inputs.append("chart_image_path")
        elif reason == "no_target_papers":
            missing_inputs.append("paper_scope")
        elif reason == "no_papers":
            missing_inputs.append("papers")
        elif reason == "missing_execution_context":
            missing_inputs.append("execution_context")
        suggested = list(observation.get("suggested_next_actions") or [])
        if not suggested:
            if status == "skipped" and reason in {"missing_task", "no_target_papers", "no_papers"}:
                suggested = ["clarify_request", "search_literature"]
            elif action_name == "answer_question":
                suggested = ["analyze_papers", "compress_context"]
            elif action_name == "search_literature":
                suggested = ["write_review", "import_papers", "answer_question"]
            elif action_name == "recommend_from_preferences":
                suggested = ["search_literature", "answer_question"]
            elif action_name == "general_answer" and metadata.get("reason") == "route_mismatch":
                suggested = [str(metadata.get("suggested_action") or "answer_question")]
            elif next_actions:
                suggested = next_actions
        state_delta = dict(observation.get("state_delta") or {})
        if action_name == "search_literature" and context.task is not None:
            state_delta.setdefault("task_id", context.task.task_id)
            state_delta.setdefault("paper_count", len(context.papers))
        if action_name == "answer_question" and context.qa_result is not None:
            state_delta.setdefault("qa_task_id", context.qa_result.task_id)
            state_delta.setdefault("paper_ids", list(context.qa_result.paper_ids))
        if action_name == "recommend_from_preferences" and context.preference_recommendation_result is not None:
            state_delta.setdefault(
                "recommended_paper_ids",
                [item.paper_id for item in context.preference_recommendation_result.recommendations],
            )
        if action_name == "general_answer":
            state_delta.setdefault("ignore_research_context", bool(metadata.get("ignore_research_context")))
        metadata["observation_envelope"] = _observation_envelope(
            progress_made=progress_made,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            missing_inputs=list(dict.fromkeys(item for item in missing_inputs if item)),
            suggested_next_actions=list(dict.fromkeys(str(item).strip() for item in suggested if str(item).strip()))[:4],
            state_delta=state_delta,
        )
        return metadata

    def action_tool_result_from_execution(self, execution_result) -> ResearchToolResult | None:
        output = execution_result.output
        if output is None:
            return None
        if isinstance(output, SupervisorActionToolOutput):
            return ResearchToolResult(
                status=output.status,
                observation=output.observation,
                metadata=dict(output.metadata),
            )
        if isinstance(output, dict):
            return ResearchToolResult(
                status=str(output.get("status") or "failed"),
                observation=str(output.get("observation") or execution_result.error_message or ""),
                metadata=dict(output.get("metadata") or {}),
            )
        return None

    def action_tool_execution_metadata(self, execution_result) -> dict[str, Any]:
        metadata = {
            "supervisor_action_call_id": execution_result.call_id,
            "supervisor_action_tool_status": execution_result.status,
            "supervisor_action_attempt_count": execution_result.attempt_count,
        }
        if execution_result.error_message:
            metadata["supervisor_action_error"] = execution_result.error_message
        return metadata

    async def execute_unified_worker(
        self,
        *,
        context: ResearchAgentToolContext,
        decision: Any,
        active_message: Any | None,
        worker_agent: str,
    ) -> UnifiedAgentResult | None:
        if active_message is None:
            return None
        registry = context.unified_agent_registry
        runtime_context = context.unified_runtime_context
        if registry is None or runtime_context is None:
            return None
        executor = registry.get(worker_agent)
        if executor is None:
            return None
        runtime_context.metadata.update(
            {
                "supervisor_tool_context": context,
                "supervisor_decision": decision,
                "supervisor_worker_agent": worker_agent,
                "supervisor_action_name": decision.action_name,
            }
        )
        task = UnifiedAgentTask.from_agent_message(
            active_message,
            preferred_skill_name=self.runtime._preferred_skill_name_for_message(
                context,
                active_message=active_message,
                worker_agent=worker_agent,
            ),
            available_tool_names=self.runtime._available_tool_names_for_agent(
                context,
                agent_name=worker_agent,
            ),
        )
        return await executor.execute(task, runtime_context)

    def research_tool_result_from_unified_result(
        self,
        unified_result: UnifiedAgentResult,
    ) -> ResearchToolResult:
        payload = dict(unified_result.payload)
        tool_metadata = payload.get("tool_metadata")
        metadata = dict(tool_metadata) if isinstance(tool_metadata, dict) else {}
        metadata.update(dict(unified_result.metadata))
        action_output = unified_result.action_output or UnifiedAgentResult.extract_action_output(
            payload=payload,
            metadata=metadata,
        )
        if action_output is not None:
            metadata.setdefault(
                UNIFIED_ACTION_OUTPUT_METADATA_KEY,
                dict(action_output),
            )
            for key, value in action_output.items():
                metadata.setdefault(key, value)
        return ResearchToolResult(
            status=unified_result.status,
            observation=str(payload.get("observation") or ""),
            metadata=metadata,
        )
