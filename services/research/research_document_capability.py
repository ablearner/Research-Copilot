from __future__ import annotations

from typing import Any

from services.research.research_knowledge_access import ResearchKnowledgeAccess


class ResearchDocumentCapability:
    """Document parsing and indexing capability used by the document specialist."""

    async def understand_document(
        self,
        *,
        graph_runtime: Any,
        file_path: str,
        document_id: str | None,
        session_id: str | None,
        metadata: dict[str, Any],
        skill_name: str | None,
        include_graph: bool,
        include_embeddings: bool,
    ) -> tuple[Any, dict[str, Any] | None]:
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        parsed_document = await knowledge_access.parse_document(
            file_path=file_path,
            document_id=document_id,
            session_id=session_id,
            metadata=metadata,
            skill_name=skill_name,
        )
        document_index_result: dict[str, Any] | None = None
        if include_graph or include_embeddings:
            index_result = await knowledge_access.index_document(
                parsed_document=parsed_document,
                charts=[],
                include_graph=include_graph,
                include_embeddings=include_embeddings,
                session_id=session_id,
                metadata=metadata,
                skill_name=skill_name,
            )
            document_index_result = index_result.model_dump(mode="json")
        return parsed_document, document_index_result
