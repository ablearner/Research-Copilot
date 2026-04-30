from __future__ import annotations

from typing import Any

from domain.schemas.api import QAResponse
from domain.schemas.document import ParsedDocument
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult
from rag_runtime.schemas import ChartUnderstandingResult, DocumentIndexResult, FusedAskResult
from tooling.schemas import GraphSummaryToolOutput, HybridRetrieveToolOutput


class ResearchKnowledgeAccess:
    """Unified access point for RAG, document, chart, and grounded QA tools."""

    def __init__(self, *, graph_runtime: Any) -> None:
        self.graph_runtime = graph_runtime

    @classmethod
    def from_runtime(cls, graph_runtime: Any) -> "ResearchKnowledgeAccess":
        existing = getattr(graph_runtime, "knowledge_access", None)
        if isinstance(existing, cls):
            return existing
        access = cls(graph_runtime=graph_runtime)
        try:
            setattr(graph_runtime, "knowledge_access", access)
        except Exception:
            pass
        return access

    async def retrieve(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        memory_hints: dict[str, Any] | None = None,
    ) -> HybridRetrieveToolOutput:
        payload = {
            "question": question,
            "document_ids": document_ids,
            "top_k": top_k,
            "filters": filters or {},
            "session_id": session_id,
            "task_id": task_id,
            "memory_hints": memory_hints or {},
        }
        tool_output = await self._execute_runtime_tool("hybrid_retrieve", payload)
        if tool_output is not None:
            return HybridRetrieveToolOutput.model_validate(tool_output)

        retrieval_tools = getattr(self.graph_runtime, "retrieval_tools", None)
        if retrieval_tools is None:
            raise RuntimeError("Graph runtime is missing hybrid_retrieve knowledge tool")
        result = await retrieval_tools.retrieve(**payload)
        return HybridRetrieveToolOutput(
            question=question,
            document_ids=document_ids,
            evidence_bundle=result.evidence_bundle,
            retrieval_result=result.retrieval_result,
            metadata=dict(getattr(result, "metadata", {}) or {}),
        )

    async def query_graph_summary(
        self,
        *,
        question: str,
        document_ids: list[str],
        top_k: int,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        memory_hints: dict[str, Any] | None = None,
    ) -> GraphSummaryToolOutput:
        payload = {
            "question": question,
            "document_ids": document_ids,
            "top_k": top_k,
            "filters": filters or {},
        }
        tool_output = await self._execute_runtime_tool("query_graph_summary", payload)
        if tool_output is not None:
            return GraphSummaryToolOutput.model_validate(tool_output)

        query_graph_summary = getattr(self.graph_runtime, "query_graph_summary", None)
        if query_graph_summary is None:
            return GraphSummaryToolOutput()
        output = await query_graph_summary(
            **payload,
            session_id=session_id,
            task_id=task_id,
            memory_hints=memory_hints or {},
        )
        return GraphSummaryToolOutput(
            hits=list(getattr(output, "hits", []) or []),
            metadata=dict(getattr(output, "metadata", {}) or {}),
        )

    async def answer_with_evidence(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result: HybridRetrievalResult | None = None,
        metadata: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        preference_context: dict[str, Any] | None = None,
        memory_hints: dict[str, Any] | None = None,
        available_tool_names: list[str] | None = None,
    ) -> QAResponse:
        research_qa_agent = getattr(self.graph_runtime, "research_qa_agent", None)
        if research_qa_agent is not None:
            return await research_qa_agent.answer(
                question=question,
                available_tool_names=available_tool_names or ["answer_with_evidence"],
                metadata=metadata or {},
                session_context=session_context or {},
                task_context=task_context or {},
                preference_context=preference_context or {},
                initial_retrieval_result=retrieval_result,
                initial_evidence_bundle=evidence_bundle,
            )

        payload = {
            "question": question,
            "evidence_bundle": evidence_bundle,
            "retrieval_result": retrieval_result,
            "metadata": metadata or {},
            "session_context": session_context or {},
            "task_context": task_context or {},
            "preference_context": preference_context or {},
            "memory_hints": memory_hints or {},
        }
        tool_output = await self._execute_runtime_tool("answer_with_evidence", payload)
        if tool_output is not None:
            return QAResponse.model_validate(tool_output)

        answer_tools = getattr(self.graph_runtime, "answer_tools", None)
        if answer_tools is None:
            raise RuntimeError("Graph runtime is missing answer_with_evidence knowledge tool")
        return await answer_tools.answer_with_evidence(**payload)

    async def parse_document(
        self,
        *,
        file_path: str,
        document_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str | None = None,
    ) -> ParsedDocument:
        payload = {
            "file_path": file_path,
            "document_id": document_id,
            "session_id": session_id,
            "metadata": metadata or {},
            "skill_name": skill_name,
        }
        tool_output = await self._execute_runtime_tool("parse_document", payload)
        if tool_output is not None:
            return ParsedDocument.model_validate(tool_output)
        return await self.graph_runtime.handle_parse_document(**payload)

    async def graph_backfill_document(
        self,
        *,
        parsed_document: ParsedDocument,
        charts: list[Any] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentIndexResult:
        payload = {
            "parsed_document": parsed_document.model_dump(mode="json"),
            "charts": charts or [],
            "session_id": session_id,
            "metadata": metadata or {},
        }
        tool_output = await self._execute_runtime_tool("graph_backfill_document", payload)
        if tool_output is not None:
            return self._coerce_document_index_result(
                tool_output,
                document_id=parsed_document.id,
            )
        result = await self.graph_runtime.handle_graph_backfill_document(
            parsed_document=parsed_document,
            charts=charts or [],
            session_id=session_id,
            metadata=metadata or {},
        )
        return self._coerce_document_index_result(
            result,
            document_id=parsed_document.id,
        )

    async def index_document(
        self,
        *,
        parsed_document: ParsedDocument,
        charts: list[Any] | None = None,
        include_graph: bool = True,
        include_embeddings: bool = True,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str | None = None,
    ) -> DocumentIndexResult:
        payload = {
            "parsed_document": parsed_document.model_dump(mode="json"),
            "charts": charts or [],
            "include_graph": include_graph,
            "include_embeddings": include_embeddings,
            "session_id": session_id,
            "metadata": metadata or {},
            "skill_name": skill_name,
        }
        tool_output = await self._execute_runtime_tool("index_document", payload)
        if tool_output is not None:
            return self._coerce_document_index_result(
                tool_output,
                document_id=parsed_document.id,
            )
        result = await self.graph_runtime.handle_index_document(
            parsed_document=parsed_document,
            charts=charts or [],
            include_graph=include_graph,
            include_embeddings=include_embeddings,
            session_id=session_id,
            metadata=metadata or {},
            skill_name=skill_name,
        )
        return self._coerce_document_index_result(
            result,
            document_id=parsed_document.id,
        )

    async def ask_document(
        self,
        *,
        question: str,
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        task_intent: str | None = None,
        metadata: dict[str, Any] | None = None,
        reasoning_style: str | None = None,
    ) -> QAResponse:
        payload = {
            "question": question,
            "doc_id": doc_id,
            "document_ids": document_ids or [],
            "top_k": top_k,
            "filters": filters or {},
            "session_id": session_id,
            "task_intent": task_intent,
            "metadata": metadata or {},
            "reasoning_style": reasoning_style,
        }
        tool_output = await self._execute_runtime_tool("ask_document", payload)
        if tool_output is not None:
            return QAResponse.model_validate(tool_output)
        return await self.graph_runtime.handle_ask_document(**payload)

    async def ask_fused(
        self,
        *,
        question: str,
        image_path: str,
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
        page_id: str | None = None,
        page_number: int = 1,
        chart_id: str | None = None,
        session_id: str | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        reasoning_style: str | None = None,
    ) -> FusedAskResult:
        payload = {
            "question": question,
            "image_path": image_path,
            "doc_id": doc_id,
            "document_ids": document_ids or [],
            "page_id": page_id,
            "page_number": page_number,
            "chart_id": chart_id,
            "session_id": session_id,
            "top_k": top_k,
            "filters": filters or {},
            "metadata": metadata or {},
            "reasoning_style": reasoning_style,
        }
        tool_output = await self._execute_runtime_tool("ask_fused", payload)
        if tool_output is not None:
            return FusedAskResult.model_validate(tool_output)
        return await self.graph_runtime.handle_ask_fused(**payload)

    async def understand_chart(
        self,
        *,
        image_path: str,
        document_id: str,
        page_id: str,
        page_number: int | None,
        chart_id: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        skill_name: str | None = None,
    ) -> ChartUnderstandingResult:
        payload = {
            "image_path": image_path,
            "document_id": document_id,
            "page_id": page_id,
            "page_number": page_number or 1,
            "chart_id": chart_id,
            "context": context or {},
            "session_id": session_id,
            "skill_name": skill_name,
        }
        tool_output = await self._execute_runtime_tool("understand_chart", payload)
        if tool_output is not None:
            return ChartUnderstandingResult.model_validate(tool_output)
        return await self.graph_runtime.handle_understand_chart(**payload)

    async def locate_chart_candidates(self, page: Any) -> list[Any]:
        document_tools = getattr(self.graph_runtime, "document_tools", None)
        if document_tools is None:
            return []
        candidates = await document_tools.locate_chart_candidates(page)
        return list(candidates or [])

    async def _execute_runtime_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> Any | None:
        tool_executor = getattr(self.graph_runtime, "tool_executor", None)
        tool_registry = getattr(self.graph_runtime, "tool_registry", None)
        if tool_executor is None or tool_registry is None:
            return None
        if tool_registry.get_tool(tool_name, include_disabled=False) is None:
            return None
        execution = await tool_executor.execute_tool_call(tool_name, tool_input)
        if execution.status == "succeeded":
            return execution.output
        raise RuntimeError(
            execution.error_message or f"knowledge tool failed: {tool_name}"
        )

    def _model_or_object_payload(self, value: Any) -> dict[str, Any]:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return dict(value)
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
            if hasattr(value, key):
                payload[key] = getattr(value, key)
        return payload or {"status": getattr(value, "status", "unknown")}

    def _coerce_document_index_result(
        self,
        value: Any,
        *,
        document_id: str,
    ) -> DocumentIndexResult:
        if isinstance(value, DocumentIndexResult):
            return value
        payload = self._model_or_object_payload(value)
        resolved_document_id = str(payload.get("document_id") or document_id).strip() or document_id
        payload["document_id"] = resolved_document_id
        payload["status"] = str(payload.get("status") or getattr(value, "status", "unknown"))
        metadata = payload.get("metadata")
        payload["metadata"] = dict(metadata or {}) if isinstance(metadata, dict) else {}
        return DocumentIndexResult.model_validate(payload)
