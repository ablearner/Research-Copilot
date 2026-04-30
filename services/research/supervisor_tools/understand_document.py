"""Understand document supervisor tool."""

from __future__ import annotations

import logging

from agents.research_supervisor_agent import ResearchSupervisorDecision
from services.research.research_knowledge_access import ResearchKnowledgeAccess
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.unified_action_adapters import (
    build_document_understanding_input,
    build_document_understanding_output,
)

logger = logging.getLogger(__name__)


class UnderstandDocumentTool:
    name = "understand_document"

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
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
