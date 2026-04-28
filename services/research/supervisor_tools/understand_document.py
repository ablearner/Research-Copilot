"""Understand document supervisor tool."""

from __future__ import annotations

import logging
from typing import Any

from agents.research_supervisor_agent import ResearchSupervisorDecision
from domain.schemas.document import ParsedDocument
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

        tool_executor = getattr(context.graph_runtime, "tool_executor", None)
        if tool_executor is not None:
            parse_execution = await tool_executor.execute_tool_call(
                "parse_document",
                {
                    "file_path": document_input.file_path,
                    "document_id": document_input.document_id,
                    "session_id": document_input.session_id,
                    "metadata": document_input.metadata,
                    "skill_name": document_input.skill_name,
                },
            )
            if parse_execution.status != "succeeded" or not isinstance(parse_execution.output, dict):
                return ResearchToolResult(
                    status=str(parse_execution.status),
                    observation=(
                        parse_execution.error_message
                        or "parse_document tool did not return a parsed document payload"
                    ),
                    metadata={"tool_name": "parse_document"},
                )
            parsed_document = ParsedDocument.model_validate(parse_execution.output)
        else:
            parsed_document = await context.graph_runtime.handle_parse_document(
                file_path=document_input.file_path,
                document_id=document_input.document_id,
                session_id=document_input.session_id,
                metadata=document_input.metadata,
                skill_name=document_input.skill_name,
            )
        context.parsed_document = parsed_document

        if document_input.include_graph or document_input.include_embeddings:
            if tool_executor is not None:
                index_execution = await tool_executor.execute_tool_call(
                    "index_document",
                    {
                        "parsed_document": parsed_document.model_dump(mode="json"),
                        "charts": [],
                        "include_graph": document_input.include_graph,
                        "include_embeddings": document_input.include_embeddings,
                        "session_id": document_input.session_id,
                        "metadata": document_input.metadata,
                        "skill_name": document_input.skill_name,
                    },
                )
                if index_execution.status == "succeeded" and isinstance(index_execution.output, dict):
                    context.document_index_result = dict(index_execution.output)
                else:
                    return ResearchToolResult(
                        status=str(index_execution.status),
                        observation=(
                            index_execution.error_message
                            or "index_document tool did not return an index payload"
                        ),
                        metadata={"tool_name": "index_document"},
                    )
            else:
                index_result = await context.graph_runtime.handle_index_document(
                    parsed_document=parsed_document,
                    charts=[],
                    include_graph=document_input.include_graph,
                    include_embeddings=document_input.include_embeddings,
                    session_id=document_input.session_id,
                    metadata=document_input.metadata,
                    skill_name=document_input.skill_name,
                )
                context.document_index_result = self._index_result_payload(index_result)

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

    def _index_result_payload(self, index_result: Any) -> dict[str, Any]:
        if hasattr(index_result, "model_dump"):
            return index_result.model_dump(mode="json")
        if isinstance(index_result, dict):
            return dict(index_result)
        payload: dict[str, Any] = {}
        for key in (
            "document_id",
            "status",
            "graph_extraction",
            "graph_index",
            "text_embedding_index",
            "page_embedding_index",
            "chart_embedding_index",
            "metadata",
        ):
            if hasattr(index_result, key):
                payload[key] = getattr(index_result, key)
        return payload or {"status": getattr(index_result, "status", "unknown")}
