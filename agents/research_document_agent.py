from __future__ import annotations

from typing import Any

from domain.schemas.unified_runtime import UnifiedAgentResult, UnifiedAgentTask
from services.research.research_document_capability import ResearchDocumentCapability
from services.research.unified_action_adapters import (
    build_document_understanding_input,
    build_document_understanding_output,
)


class ResearchDocumentAgent:
    """Supervisor-delegated specialist for document understanding."""

    name = "ResearchDocumentAgent"

    def __init__(self, *, capability: ResearchDocumentCapability | None = None) -> None:
        self.capability = capability or ResearchDocumentCapability()

    async def execute(self, task: UnifiedAgentTask, runtime_context: Any) -> UnifiedAgentResult:
        supervisor_context = runtime_context.metadata.get("supervisor_tool_context")
        decision = runtime_context.metadata.get("supervisor_decision")
        if supervisor_context is None or decision is None:
            return self._unified_result(
                task=task,
                status="failed",
                observation="missing supervisor runtime context for ResearchDocumentAgent",
                metadata={"reason": "missing_supervisor_runtime_context"},
            )

        supervisor_context.document_attempted = True
        document_input = build_document_understanding_input(context=supervisor_context, decision=decision)
        if not document_input.file_path:
            return self._unified_result(
                task=task,
                status="skipped",
                observation="no document_file_path was provided for document understanding",
                metadata={"reason": "missing_document_file_path"},
            )

        try:
            parsed_document, document_index_result = await self.capability.understand_document(
                graph_runtime=supervisor_context.graph_runtime,
                file_path=document_input.file_path,
                document_id=document_input.document_id,
                session_id=document_input.session_id,
                metadata=document_input.metadata,
                skill_name=document_input.skill_name,
                include_graph=document_input.include_graph,
                include_embeddings=document_input.include_embeddings,
            )
        except RuntimeError as exc:
            return self._unified_result(
                task=task,
                status="failed",
                observation=str(exc),
                metadata={"tool_name": "document_understanding"},
            )

        supervisor_context.parsed_document = parsed_document
        supervisor_context.document_index_result = document_index_result
        output = build_document_understanding_output(
            parsed_document=parsed_document,
            document_index_result=document_index_result,
        )
        metadata = output.to_metadata()
        metadata.update(
            {
                "executed_by": self.name,
                "document_execution_path": "research_document_agent",
            }
        )
        return self._unified_result(
            task=task,
            status="succeeded",
            observation=(
                f"document understood; document_id={parsed_document.id}; "
                f"pages={len(parsed_document.pages)}; indexed={bool(document_index_result)}"
            ),
            metadata=metadata,
        )

    def _unified_result(
        self,
        *,
        task: UnifiedAgentTask,
        status: str,
        observation: str,
        metadata: dict[str, Any],
    ) -> UnifiedAgentResult:
        return UnifiedAgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            task_type=task.task_type,
            status=status,  # type: ignore[arg-type]
            instruction=task.instruction,
            payload={
                "observation": observation,
                "tool_metadata": dict(metadata),
            },
            context_slice=task.context_slice,
            priority=task.priority,
            expected_output_schema=task.expected_output_schema,
            depends_on=task.depends_on,
            retry_count=task.retry_count,
            action_output=(
                dict(metadata)
                if UnifiedAgentResult.is_action_output_payload(metadata)
                else None
            ),
            metadata={
                "execution_engine": "unified_agent_registry",
                "execution_adapter": "research_document_agent",
                "delegate_type": self.__class__.__name__,
            },
        )
