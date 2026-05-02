from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from tools.research.document_capability import ResearchDocumentCapability

if TYPE_CHECKING:
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )

logger = logging.getLogger(__name__)


class ResearchDocumentAgent:
    """Supervisor-delegated specialist for document understanding."""

    name = "ResearchDocumentAgent"

    def __init__(self, *, capability: ResearchDocumentCapability | None = None) -> None:
        self.capability = capability or ResearchDocumentCapability()

    # ------------------------------------------------------------------
    # New unified entry point (SpecialistAgent protocol)
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
    ) -> ResearchToolResult:
        from tools.research.knowledge_access import ResearchKnowledgeAccess
        from runtime.research.agent_protocol.base import ResearchToolResult
        from runtime.research.unified_action_adapters import (
            build_document_understanding_input,
            build_document_understanding_output,
        )

        context.document_attempted = True
        document_input = build_document_understanding_input(context=context, decision=decision)
        if not document_input.file_path:
            return ResearchToolResult(
                status="skipped",
                observation="no document_file_path was provided for document understanding",
                metadata={"reason": "missing_document_file_path"},
            )

        knowledge_access = context.knowledge_access or ResearchKnowledgeAccess.from_runtime(context.graph_runtime)
        try:
            parsed_document = await knowledge_access.parse_document(
                file_path=document_input.file_path,
                document_id=document_input.document_id,
                session_id=document_input.session_id,
                metadata=document_input.metadata,
                skill_name=document_input.skill_name,
            )
        except RuntimeError as exc:
            return ResearchToolResult(
                status="failed",
                observation=str(exc),
                metadata={"tool_name": "parse_document"},
            )
        context.parsed_document = parsed_document

        if document_input.include_graph or document_input.include_embeddings:
            try:
                index_result = await knowledge_access.index_document(
                    parsed_document=parsed_document,
                    charts=[],
                    include_graph=document_input.include_graph,
                    include_embeddings=document_input.include_embeddings,
                    session_id=document_input.session_id,
                    metadata=document_input.metadata,
                    skill_name=document_input.skill_name,
                )
                context.document_index_result = index_result.model_dump(mode="json")
            except RuntimeError as exc:
                return ResearchToolResult(
                    status="failed",
                    observation=str(exc),
                    metadata={"tool_name": "index_document"},
                )

        output = build_document_understanding_output(
            parsed_document=parsed_document,
            document_index_result=context.document_index_result,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"document understood; document_id={parsed_document.id}; "
                f"pages={len(parsed_document.pages)}; indexed={bool(context.document_index_result)}"
            ),
            metadata=output.to_metadata(),
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
