from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

from core.prompt_resolver import PromptResolver
from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import ParsedDocument
from domain.schemas.evidence import Evidence, EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit
from rag_runtime.checkpoint import build_checkpointer
from rag_runtime.memory import GraphSessionMemory
from rag_runtime.schemas import (
    ChartUnderstandingResult,
    DocumentIndexResult,
    FusedAskResult,
    GraphTaskRequest,
    GraphTaskResult,
)
from rag_runtime.state import ChartDocRAGState
from planners.function_calling import AnswerValidationPlanner, RetrievalPlan, RetrievalPlanner
from reasoning.strategies import ReasoningStrategySet
from retrieval.evidence_builder import build_evidence_bundle
from skills.base import SkillSpec
from skills.registry import SkillRegistry
from mcp.client.registry import MCPClientRegistry
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from tooling.schemas import GraphSummaryToolOutput, ToolCallTrace, ToolExecutionResult
from tools.chart_toolkit import explain_chart, visible_chart_text


logger = logging.getLogger(__name__)
_MAX_ASK_EVIDENCES = 20
_MAX_ASK_HITS = 10


class RagRuntime:
    """Tool-first RAG runtime facade.

    `GraphRuntime` remains as a compatibility alias while the codebase
    transitions away from the historical graph-centric naming.
    """

    def __init__(
        self,
        document_tools: Any,
        chart_tools: Any,
        graph_extraction_tools: Any,
        retrieval_tools: Any,
        answer_tools: Any,
        graph_index_service: Any,
        embedding_index_service: Any,
        *,
        llm_adapter: Any | None = None,
        checkpointer: Any | None = None,
        session_memory: GraphSessionMemory | None = None,
        prompt_resolver: PromptResolver | None = None,
        skill_registry: SkillRegistry | None = None,
        tool_registry: ToolRegistry | None = None,
        tool_executor: ToolExecutor | None = None,
        mcp_client_registry: MCPClientRegistry | None = None,
        reasoning_strategies: ReasoningStrategySet | None = None,
        cot_reasoning_agent: Any | None = None,
        plan_and_solve_reasoning_agent: Any | None = None,
        react_reasoning_agent: Any | None = None,
        graph_execution_mode: str = "tool",
        **_: Any,
    ) -> None:
        if any(
            component is None
            for component in [
                document_tools,
                chart_tools,
                graph_extraction_tools,
                retrieval_tools,
                answer_tools,
                graph_index_service,
                embedding_index_service,
            ]
        ):
            raise TypeError(
                "RagRuntime requires document/chart/graph_extraction/retrieval/answer capabilities and index services"
            )

        self.document_tools = document_tools
        self.chart_tools = chart_tools
        self.graph_extraction_tools = graph_extraction_tools
        self.retrieval_tools = retrieval_tools
        self.answer_tools = answer_tools
        self.graph_index_service = graph_index_service
        self.embedding_index_service = embedding_index_service
        self.llm_adapter = llm_adapter or getattr(answer_tools, "llm_adapter", None)
        self.checkpointer = checkpointer or build_checkpointer()
        self.session_memory = session_memory or GraphSessionMemory()
        self.prompt_resolver = (
            prompt_resolver or getattr(answer_tools, "prompt_resolver", None) or PromptResolver()
        )
        self.skill_registry = skill_registry or SkillRegistry()
        self.tool_registry = tool_registry or ToolRegistry()
        self.tool_executor = tool_executor or ToolExecutor(self.tool_registry)
        self.external_tool_registry = mcp_client_registry or MCPClientRegistry()
        self.reasoning_strategies = reasoning_strategies or ReasoningStrategySet(
            query_planning=plan_and_solve_reasoning_agent,
            answer_synthesis=cot_reasoning_agent,
            tool_reasoning=react_reasoning_agent,
        )
        self.cot_reasoning_agent = (
            cot_reasoning_agent
            or self.reasoning_strategies.cot_reasoning_agent
        )
        self.plan_and_solve_reasoning_agent = (
            plan_and_solve_reasoning_agent
            or self.reasoning_strategies.plan_and_solve_reasoning_agent
        )
        self.react_reasoning_agent = (
            react_reasoning_agent
            or self.reasoning_strategies.react_reasoning_agent
        )
        self.graph_execution_mode = graph_execution_mode
        self.router_planner_agent = None
        self.retrieval_planner_agent = RetrievalPlanner(self.llm_adapter)
        self.validation_agent = AnswerValidationPlanner(self.llm_adapter)
        self.graph_builder = None
        self.graph = None

    @property
    def mcp_client_registry(self) -> MCPClientRegistry | None:
        """Compatibility alias for the historical MCP-specific field name."""
        return self.external_tool_registry

    @mcp_client_registry.setter
    def mcp_client_registry(self, value: MCPClientRegistry | None) -> None:
        self.external_tool_registry = value

    async def invoke(self, state: ChartDocRAGState) -> ChartDocRAGState:
        started_at = time.perf_counter()
        request_id = state.get("request_id") or f"req_{uuid4().hex}"
        session_id = state.get("session_id")
        thread_id = state.get("thread_id") or session_id or request_id
        try:
            memory_snapshot = self.session_memory.load(session_id)
        except Exception:
            logger.exception(
                "Session memory load failed; continuing without memory",
                extra={"session_id": session_id},
            )
            memory_snapshot = None

        prepared_state = self._prepare_state(
            {
                **state,
                "request_id": request_id,
                "thread_id": thread_id,
                "session_memory": self.session_memory.as_prompt_context(memory_snapshot),
                "retrieval_attempt": state.get("retrieval_attempt", 0),
                "max_retrieval_attempts": state.get("max_retrieval_attempts", 1),
            }
        )
        result = await self._run_pipeline(prepared_state)

        total_latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        result["metadata"] = {
            **result.get("metadata", {}),
            "trace_id": request_id,
            "thread_id": thread_id,
            "runtime_total_latency_ms": total_latency_ms,
            "tool_trace_count": len(result.get("tool_traces", [])),
            "warning_count": len(result.get("warnings", [])),
            "runtime_engine": "tool_runtime",
        }
        try:
            self.session_memory.update_from_state(result)
        except Exception:
            logger.exception(
                "Session memory update failed; returning runtime result anyway",
                extra={"session_id": session_id, "request_id": request_id},
            )
        return result

    async def _run_pipeline(self, state: ChartDocRAGState) -> ChartDocRAGState:
        task_type = state.get("task_type")
        if task_type == "parse":
            return await self._run_parse_pipeline(state)
        if task_type == "index":
            return await self._run_index_pipeline(state)
        if task_type == "chart_understand":
            return await self._run_chart_understand_pipeline(state)
        if task_type == "ask":
            if state.get("task_intent") == "ask_fused" and state.get("image_path"):
                return await self._run_fused_ask_pipeline(state)
            return await self._run_ask_pipeline(state)
        return self._merge_state(
            state,
            {
                "errors": [f"Unsupported task_type: {task_type}"],
                "warnings": [f"RagRuntime could not route task_type={task_type!r}"],
            },
        )

    def _prepare_state(self, state: ChartDocRAGState) -> ChartDocRAGState:
        return {
            "vector_hits": [],
            "graph_hits": [],
            "summary_hits": [],
            "graph_summary_hits": [],
            "warnings": [],
            "messages": [],
            "tool_traces": [],
            "errors": [],
            "metadata": {},
            "filters": {},
            "reasoning_summary": {},
            "react_trace": [],
            "retrieval_mode": "hybrid",
            "top_k": 10,
            "retrieval_attempt": 0,
            "max_retrieval_attempts": 1,
            **state,
        }

    async def handle(self, request: GraphTaskRequest) -> GraphTaskResult:
        if request.task_type == "function_call":
            params = request.params
            result = await self.handle_function_call(
                tool_name=str(params.get("tool_name") or ""),
                arguments=params.get("arguments") or {},
            )
            return GraphTaskResult(
                task_type=request.task_type,
                status="failed" if result.status != "succeeded" else "succeeded",
                trace_id=request.trace_id,
                output=result.output,
                error_message=result.error_message,
                metadata={
                    "runtime_engine": "tool_runtime",
                    "tool_name": result.tool_name,
                    "tool_status": result.status,
                    "tool_trace": result.trace.model_dump(mode="json"),
                },
            )
        if request.task_type == "ask_fused":
            params = request.params
            output = await self.handle_ask_fused(
                question=params.get("question") or "",
                image_path=params.get("image_path") or "",
                doc_id=params.get("document_id") or params.get("doc_id"),
                document_ids=params.get("document_ids"),
                page_id=params.get("page_id"),
                page_number=params.get("page_number", 1),
                chart_id=params.get("chart_id"),
                session_id=params.get("session_id"),
                top_k=params.get("top_k", 10),
                filters=params.get("filters"),
                metadata=params.get("metadata") or params.get("context"),
                skill_name=params.get("skill_name"),
                reasoning_style=params.get("reasoning_style"),
            )
            return GraphTaskResult(
                task_type=request.task_type,
                status="succeeded",
                trace_id=request.trace_id,
                output=output,
                metadata={"runtime_engine": "tool_runtime", **output.metadata},
            )
        mapped = self._request_to_state(request)
        result = await self.invoke(mapped)
        return GraphTaskResult(
            task_type=request.task_type,
            status="failed" if result.get("errors") else "succeeded",
            trace_id=request.trace_id or result.get("request_id"),
            output=self._state_output(result),
            error_message="; ".join(result.get("errors", [])) or None,
            metadata={"runtime_engine": "tool_runtime", **result.get("metadata", {})},
        )

    async def handle_parse_document(
        self,
        file_path: str,
        document_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str | None = None,
        **_: Any,
    ) -> ParsedDocument:
        skill_context = self.resolve_skill_context(
            task_type="parse",
            preferred_skill_name=skill_name,
        )
        state = await self.invoke(
            {
                "task_type": "parse",
                "file_path": file_path,
                "document_id": document_id,
                "session_id": session_id,
                "user_input": file_path,
                "document_ids": [document_id] if document_id else [],
                "metadata": {
                    **(metadata or {}),
                    **({"skill_name": skill_name} if skill_name else {}),
                },
                "selected_skill": skill_context,
            }
        )
        return state["parsed_document"]

    async def handle_index_document(
        self,
        parsed_document: ParsedDocument,
        charts: list[ChartSchema] | None = None,
        include_graph: bool = True,
        include_embeddings: bool = True,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str | None = None,
        **_: Any,
    ) -> DocumentIndexResult:
        skill_context = self.resolve_skill_context(
            task_type="index",
            preferred_skill_name=skill_name,
        )
        metadata = {
            **(metadata or {}),
            "charts": charts or [],
            "include_graph": include_graph,
            "include_embeddings": include_embeddings,
            **({"skill_name": skill_name} if skill_name else {}),
        }
        state = await self.invoke(
            {
                "task_type": "index",
                "document_id": parsed_document.id,
                "document_ids": [parsed_document.id],
                "session_id": session_id,
                "parsed_document": parsed_document,
                "charts": charts or [],
                "include_graph": include_graph,
                "include_embeddings": include_embeddings,
                "user_input": parsed_document.filename,
                "metadata": metadata,
                "selected_skill": skill_context,
            }
        )
        graph_index_payload = state.get("metadata", {}).get("graph_index")
        return DocumentIndexResult(
            document_id=parsed_document.id,
            graph_extraction=state.get("graph_extraction_result"),
            graph_index=self._model_from_payload(graph_index_payload),
            text_embedding_index=self._model_from_payload(
                state.get("metadata", {}).get("text_embedding_index")
            ),
            page_embedding_index=self._model_from_payload(
                state.get("metadata", {}).get("page_embedding_index")
            ),
            chart_embedding_index=self._model_from_payload(
                state.get("metadata", {}).get("chart_embedding_index")
            ),
            status="failed" if state.get("errors") else "succeeded",
            metadata={"runtime_engine": "tool_runtime", **state.get("metadata", {})},
        )

    async def handle_graph_backfill_document(
        self,
        parsed_document: ParsedDocument,
        *,
        charts: list[ChartSchema] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentIndexResult:
        graph_result = None
        graph_index = None
        try:
            text_blocks = [block for page in parsed_document.pages for block in page.text_blocks]
            graph_result = await self.graph_extraction_tools.extract_from_text_blocks(
                document_id=parsed_document.id,
                text_blocks=text_blocks,
                page_summaries=[],
            )
            graph_index = await self.graph_index_service.index_graph_result(graph_result)
            status = "succeeded"
            error_message = None
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_message = f"{exc.__class__.__name__}: {exc}"
        return DocumentIndexResult(
            document_id=parsed_document.id,
            graph_extraction=graph_result,
            graph_index=graph_index,
            status=status,
            metadata={
                "runtime_engine": "tool_runtime",
                "index_mode": "graph_backfill",
                "session_id": session_id,
                **(metadata or {}),
                **({"error_message": error_message} if error_message else {}),
            },
        )

    async def handle_ask_document(
        self,
        question: str,
        doc_id: str | None = None,
        document_ids: list[str] | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        task_intent: str | None = None,
        metadata: dict[str, Any] | None = None,
        skill_name: str | None = None,
        reasoning_style: str | None = None,
        **_: Any,
    ) -> QAResponse:
        resolved_document_ids = document_ids or ([doc_id] if doc_id else [])
        skill_context = self.resolve_skill_context(
            task_type=task_intent or "ask_document",
            preferred_skill_name=skill_name,
        )
        if self._is_chart_like_question(question):
            visual_anchor = await self._resolve_visual_anchor(
                question=question,
                doc_id=doc_id,
                document_ids=resolved_document_ids,
                top_k=top_k,
                filters=filters or {},
                session_id=session_id,
                skill_context=skill_context,
            )
            if visual_anchor is not None:
                fused_result = await self.handle_ask_fused(
                    question=question,
                    image_path=visual_anchor["image_path"],
                    doc_id=doc_id,
                    document_ids=resolved_document_ids,
                    page_id=visual_anchor.get("page_id"),
                    page_number=visual_anchor.get("page_number", 1),
                    chart_id=visual_anchor.get("chart_id"),
                    session_id=session_id,
                    top_k=top_k,
                    filters=filters,
                    metadata={
                        **(metadata or {}),
                        "auto_fused": True,
                        "auto_fused_reason": "chart_like_question",
                        "visual_anchor": visual_anchor,
                    },
                    skill_name=skill_name,
                    reasoning_style=reasoning_style,
                )
                return fused_result.qa

        state = await self.invoke(
            {
                "task_type": "ask",
                "user_input": question,
                "document_id": doc_id,
                "document_ids": resolved_document_ids,
                "session_id": session_id,
                "task_intent": task_intent or "ask_document",
                "top_k": top_k,
                "filters": filters or {},
                "metadata": {
                    **(metadata or {}),
                    **({"reasoning_style": reasoning_style} if reasoning_style else {}),
                },
                "retrieval_mode": "hybrid",
                "max_retrieval_attempts": 1,
                "selected_skill": skill_context,
            }
        )
        answer = state["final_answer"]
        if isinstance(answer, QAResponse):
            answer = self._ensure_actionable_answer(
                answer,
                question=question,
                evidence_bundle=state.get("evidence_bundle") or answer.evidence_bundle,
                retrieval_result=answer.retrieval_result,
                metadata={
                    "runtime_engine": "tool_runtime",
                    **state.get("metadata", {}),
                    "reasoning_summary": state.get("reasoning_summary", {}),
                    "warnings": state.get("warnings", []),
                    "tool_traces": state.get("tool_traces", []),
                    "react_trace": state.get("react_trace", []),
                },
            )
            return self._compact_qa_response(
                answer,
                document_ids=state.get("document_ids") or resolved_document_ids,
            ).model_copy(
                update={
                    "metadata": {
                        **answer.metadata,
                        "runtime_engine": "tool_runtime",
                        **state.get("metadata", {}),
                        "reasoning_summary": state.get("reasoning_summary", {}),
                        "warnings": state.get("warnings", []),
                        "tool_traces": state.get("tool_traces", []),
                        "react_trace": state.get("react_trace", []),
                        "evidence_mix": {
                            "vector_hits": len(state.get("vector_hits", [])),
                            "graph_hits": len(state.get("graph_hits", [])),
                            "graph_summary_hits": len(state.get("graph_summary_hits", [])),
                            "evidence_count": len(
                                (state.get("evidence_bundle") or answer.evidence_bundle).evidences
                            ),
                        },
                    }
                }
            )
        return QAResponse(
            answer=str(answer.get("answer", "证据不足")) if isinstance(answer, dict) else "证据不足",
            question=question,
            evidence_bundle=state.get("evidence_bundle") or answer.get("evidence_bundle"),
            confidence=state.get("confidence"),
            metadata={
                "runtime_engine": "tool_runtime",
                **state.get("metadata", {}),
                "reasoning_summary": state.get("reasoning_summary", {}),
                "warnings": state.get("warnings", []),
                "tool_traces": state.get("tool_traces", []),
                "react_trace": state.get("react_trace", []),
                "evidence_mix": {
                    "vector_hits": len(state.get("vector_hits", [])),
                    "graph_hits": len(state.get("graph_hits", [])),
                    "graph_summary_hits": len(state.get("graph_summary_hits", [])),
                    "evidence_count": len((state.get("evidence_bundle") or {}).evidences)
                    if state.get("evidence_bundle")
                    else 0,
                },
            },
        )

    async def _resolve_visual_anchor(
        self,
        *,
        question: str,
        doc_id: str | None,
        document_ids: list[str],
        top_k: int,
        filters: dict[str, Any],
        session_id: str | None,
        skill_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for retrieval_filters in (
            {**filters, "retrieval_mode": "vector", "modalities": ["chart", "page"]},
            {**filters, "retrieval_mode": "vector", "modalities": ["page"]},
        ):
            try:
                result = await self.retrieval_tools.retrieve(
                    question=question,
                    doc_id=doc_id,
                    document_ids=document_ids,
                    top_k=max(3, min(top_k, 6)),
                    filters=retrieval_filters,
                    session_id=session_id,
                    task_id=f"visual_anchor_{uuid4().hex}",
                    memory_hints={},
                    skill_context=skill_context,
                )
            except Exception:
                logger.exception("Failed to resolve visual anchor for chart-like question")
                continue
            candidate = self._select_visual_anchor_from_hits(result.retrieval_result.hits)
            if candidate is not None:
                candidates.append(candidate)
        return candidates[0] if candidates else None

    def _select_visual_anchor_from_hits(self, hits: list[Any]) -> dict[str, Any] | None:
        for hit in hits:
            metadata = getattr(hit, "metadata", {}) or {}
            image_path = metadata.get("uri") or metadata.get("image_uri") or metadata.get("image_path")
            if not image_path:
                continue
            page_id = metadata.get("page_id")
            page_number = self._coerce_page_number(metadata.get("page_number")) or 1
            chart_id = hit.source_id if getattr(hit, "source_type", None) == "chart" else None
            if chart_id is None:
                chart_id = f"chart_{page_id or getattr(hit, 'id', 'auto')}"
            return {
                "image_path": str(image_path),
                "page_id": str(page_id) if page_id else None,
                "page_number": page_number,
                "chart_id": str(chart_id),
                "source_type": getattr(hit, "source_type", None),
                "source_id": getattr(hit, "source_id", None),
            }
        return None

    def _coerce_page_number(self, value: Any) -> int | None:
        try:
            page_number = int(value)
        except (TypeError, ValueError):
            return None
        return page_number if page_number >= 1 else None

    def _is_chart_like_question(self, question: str) -> bool:
        normalized = question.strip().lower()
        if not normalized:
            return False
        markers = (
            "this graph",
            "the graph",
            "this chart",
            "the chart",
            "this figure",
            "the figure",
            "this plot",
            "the plot",
            "bar chart",
            "line chart",
            "scatter plot",
            "pie chart",
            "x-axis",
            "y-axis",
            "这张图",
            "这个图",
            "该图",
            "图表",
            "图中",
            "图里",
            "柱状图",
            "折线图",
            "散点图",
            "饼图",
        )
        return any(marker in normalized for marker in markers)

    def _ensure_actionable_answer(
        self,
        answer: QAResponse,
        *,
        question: str,
        evidence_bundle: Any,
        retrieval_result: Any,
        metadata: dict[str, Any],
    ) -> QAResponse:
        bundle = (
            evidence_bundle
            if isinstance(evidence_bundle, EvidenceBundle)
            else getattr(answer, "evidence_bundle", None)
        )
        if bundle is None or not bundle.evidences:
            return answer
        answer_text = (answer.answer or "").strip().lower()
        if "证据不足" not in answer_text and "insufficient evidence" not in answer_text:
            return answer

        snippets = [
            (evidence.snippet or "").strip()
            for evidence in bundle.evidences[:3]
            if (evidence.snippet or "").strip()
        ]
        if not snippets:
            return answer

        fallback_lines = ["根据当前检索到的证据，可以确认以下信息："]
        for index, snippet in enumerate(snippets, start=1):
            fallback_lines.append(f"{index}. {snippet[:240]}")
        fallback_lines.append("以上回答基于当前证据自动整理，如需更精确结论可继续追问。")
        return answer.model_copy(
            update={
                "answer": "\n".join(fallback_lines),
                "question": question,
                "evidence_bundle": bundle,
                "retrieval_result": retrieval_result,
                "confidence": min(answer.confidence if answer.confidence is not None else 0.2, 0.62),
                "metadata": {
                    **answer.metadata,
                    **metadata,
                    "answered_by": "RagRuntimeFallback",
                    "fallback": True,
                    "fallback_reason": "insufficient_model_answer",
                    "evidence_count": len(bundle.evidences),
                },
            }
        )

    async def handle_ask_fused(
        self,
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
        skill_name: str | None = None,
        reasoning_style: str | None = None,
        **_: Any,
    ) -> FusedAskResult:
        resolved_document_ids = document_ids or ([doc_id] if doc_id else [])
        skill_context = self.resolve_skill_context(
            task_type="ask_document",
            preferred_skill_name=skill_name,
        )
        state = await self.invoke(
            {
                "task_type": "ask",
                "user_input": question,
                "document_id": doc_id,
                "document_ids": resolved_document_ids,
                "session_id": session_id,
                "task_intent": "ask_fused",
                "image_path": image_path,
                "page_id": page_id,
                "page_number": page_number,
                "chart_id": chart_id,
                "top_k": top_k,
                "filters": filters or {},
                "metadata": {
                    **(metadata or {}),
                    **({"reasoning_style": reasoning_style} if reasoning_style else {}),
                },
                "retrieval_mode": "hybrid",
                "max_retrieval_attempts": 1,
                "selected_skill": skill_context,
            }
        )
        answer = state["final_answer"]
        evidence_bundle = state.get("evidence_bundle") or (
            answer.evidence_bundle if isinstance(answer, QAResponse) else None
        )
        chart_evidence_count = (
            sum(1 for evidence in evidence_bundle.evidences if evidence.source_type == "chart")
            if evidence_bundle is not None
            else 0
        )
        response_metadata = {
            "runtime_engine": "tool_runtime",
            **state.get("metadata", {}),
            "reasoning_summary": state.get("reasoning_summary", {}),
            "warnings": state.get("warnings", []),
            "tool_traces": state.get("tool_traces", []),
            "react_trace": state.get("react_trace", []),
            "task_intent": state.get("task_intent"),
            "fused": state.get("task_intent") == "ask_fused",
            "chart_answer": state.get("chart_answer"),
            "chart_confidence": state.get("chart_confidence"),
            "evidence_mix": {
                "vector_hits": len(state.get("vector_hits", [])),
                "graph_hits": len(state.get("graph_hits", [])),
                "graph_summary_hits": len(state.get("graph_summary_hits", [])),
                "chart_evidence_count": chart_evidence_count,
                "evidence_count": len(evidence_bundle.evidences) if evidence_bundle is not None else 0,
            },
        }
        if isinstance(answer, QAResponse):
            qa = self._compact_qa_response(
                answer,
                document_ids=resolved_document_ids,
            ).model_copy(update={"metadata": {**answer.metadata, **response_metadata}})
        else:
            qa = self._compact_qa_response(
                QAResponse(
                    answer=str(answer.get("answer", "证据不足")) if isinstance(answer, dict) else "证据不足",
                    question=question,
                    evidence_bundle=state.get("evidence_bundle") or answer.get("evidence_bundle"),
                    confidence=state.get("confidence"),
                    metadata=response_metadata,
                ),
                document_ids=resolved_document_ids,
            )
        return FusedAskResult(
            qa=qa,
            chart_answer=state.get("chart_answer"),
            chart_confidence=state.get("chart_confidence"),
            metadata=response_metadata,
        )

    def _compact_qa_response(self, qa: QAResponse, *, document_ids: list[str]) -> QAResponse:
        allowed_document_ids = {doc_id for doc_id in document_ids if doc_id}

        def keep_evidence(document_id: str | None) -> bool:
            return not allowed_document_ids or (document_id in allowed_document_ids)

        evidence_bundle = qa.evidence_bundle.model_copy(
            update={
                "evidences": [
                    evidence for evidence in qa.evidence_bundle.evidences if keep_evidence(evidence.document_id)
                ][:_MAX_ASK_EVIDENCES],
            }
        )
        retrieval_result = qa.retrieval_result
        if retrieval_result is not None:
            retrieval_result = retrieval_result.model_copy(
                update={
                    "hits": [hit for hit in retrieval_result.hits if keep_evidence(hit.document_id)][
                        :_MAX_ASK_HITS
                    ],
                }
            )
        return qa.model_copy(update={"evidence_bundle": evidence_bundle, "retrieval_result": retrieval_result})

    async def handle_understand_chart(
        self,
        image_path: str,
        document_id: str,
        page_id: str,
        page_number: int,
        chart_id: str,
        session_id: str | None = None,
        context: dict[str, Any] | None = None,
        skill_name: str | None = None,
        **_: Any,
    ) -> ChartUnderstandingResult:
        skill_context = self.resolve_skill_context(
            task_type="understand_chart",
            preferred_skill_name=skill_name,
        )
        state = await self.invoke(
            {
                "task_type": "chart_understand",
                "user_input": image_path,
                "image_path": image_path,
                "document_id": document_id,
                "document_ids": [document_id],
                "session_id": session_id,
                "page_id": page_id,
                "page_number": page_number,
                "chart_id": chart_id,
                "metadata": context or {},
                "selected_skill": skill_context,
            }
        )
        chart_result = state["chart_result"]
        return ChartUnderstandingResult(
            chart=chart_result["chart"],
            graph_text=chart_result["graph_text"],
            metadata={
                "runtime_engine": "tool_runtime",
                **chart_result.get("metadata", {}),
                **({"skill_name": skill_name} if skill_name else {}),
            },
        )

    async def query_graph_summary(
        self,
        question: str,
        document_ids: list[str] | None = None,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        session_id: str | None = None,
        task_id: str | None = None,
        memory_hints: dict[str, Any] | None = None,
        skill_context: dict[str, Any] | None = None,
        **_: Any,
    ) -> GraphSummaryToolOutput:
        retrieval = await self.retrieval_tools.retrieve(
            question=question,
            document_ids=document_ids or [],
            top_k=top_k,
            filters={**(filters or {}), "retrieval_mode": "graphrag_summary"},
            session_id=session_id,
            task_id=task_id,
            memory_hints=memory_hints or {},
            skill_context=skill_context,
        )
        return GraphSummaryToolOutput(
            hits=retrieval.retrieval_result.hits,
            metadata={
                "question": question,
                "document_ids": document_ids or [],
                "hit_count": len(retrieval.retrieval_result.hits),
                "source": "graph_summary_tool",
            },
        )

    async def handle_function_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        **_: Any,
    ) -> ToolExecutionResult:
        if self.tool_executor is None:
            raise RuntimeError("Tool executor is not configured")
        local_tool = self.tool_registry.get_tool(tool_name, include_disabled=True) if self.tool_registry else None
        if local_tool is None and self.external_tool_registry is not None:
            external_result = await self.external_tool_registry.call_tool(
                tool_name=tool_name,
                arguments=arguments or {},
            )
            return ToolExecutionResult(
                call_id=external_result.call_id,
                tool_name=external_result.tool_name,
                status=self._map_mcp_status_to_tool_status(external_result.status),
                output=external_result.output,
                error_message=external_result.error_message,
                validation_passed=external_result.status != "invalid_input",
                trace=self._build_external_tool_trace(
                    tool_name=external_result.tool_name,
                    call_id=external_result.call_id,
                    tool_input=arguments or {},
                    output=external_result.output,
                    status=self._map_mcp_status_to_tool_status(external_result.status),
                    error_message=external_result.error_message,
                    latency_ms=external_result.latency_ms,
                    server_name=external_result.server_name,
                ),
            )
        return await self.tool_executor.execute_tool_call(
            tool_name=tool_name,
            tool_input=arguments or {},
        )

    def list_function_tools(
        self,
        *,
        skill_context: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        names: list[str] | None = None,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        if self.tool_registry is None:
            return []
        tool_specs = self.tool_registry.filter_tools(
            tags=tags,
            enabled_only=not include_disabled,
            names=names,
            skill_context=skill_context,
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema.model_json_schema(),
                },
            }
            for tool in tool_specs
        ]

    async def list_external_function_tools(
        self,
        *,
        task_type: str = "function_call",
        preferred_skill_name: str | None = None,
        skill_context: dict[str, Any] | None = None,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        if self.skill_registry is None or self.external_tool_registry is None:
            return []
        resolved_skill_name = preferred_skill_name
        if resolved_skill_name is None and isinstance(skill_context, dict):
            candidate = skill_context.get("name")
            if isinstance(candidate, str) and candidate.strip():
                resolved_skill_name = candidate
        allowed_names = self.skill_registry.allowed_external_mcp_tools(
            task_type=task_type,
            preferred_skill_name=resolved_skill_name,
        )
        if not allowed_names:
            return []
        tools = await self.external_tool_registry.discover_tools()
        filtered_tools = [
            tool
            for tool in tools
            if tool.name in allowed_names and (include_disabled or tool.enabled)
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
                "server_name": tool.server_name,
                "source": tool.source,
                "enabled": tool.enabled,
                "tags": tool.tags,
                "output_schema": tool.output_schema,
            }
            for tool in filtered_tools
        ]

    def resolve_skill_context(
        self,
        *,
        task_type: str,
        preferred_skill_name: str | None = None,
    ) -> dict[str, Any] | None:
        if self.skill_registry is None:
            return None
        try:
            skill = self.skill_registry.select_skill_for_task(
                task_type=task_type,
                preferred_skill_name=preferred_skill_name,
            )
        except Exception:
            logger.exception(
                "Failed to resolve skill context",
                extra={"task_type": task_type, "preferred_skill_name": preferred_skill_name},
            )
            return None
        return self._skill_to_context(skill, task_type=task_type)

    def _skill_to_context(self, skill: SkillSpec, *, task_type: str) -> dict[str, Any]:
        return {
            "name": skill.name,
            "description": skill.description,
            "task_type": task_type,
            "prompt_set": skill.prompt_set.model_dump(mode="json"),
            "preferred_tools": list(skill.preferred_tools),
            "retrieval_policy": skill.retrieval_policy.model_dump(mode="json"),
            "memory_policy": skill.memory_policy.model_dump(mode="json"),
            "output_style": skill.output_style.model_dump(mode="json"),
            "enabled": skill.enabled,
            "metadata": dict(skill.metadata),
        }

    def _map_mcp_status_to_tool_status(self, status: str) -> str:
        if status == "invalid_input":
            return "validation_error"
        if status in {"succeeded", "failed", "not_found", "disabled"}:
            return status
        return "failed"

    def _build_external_tool_trace(
        self,
        *,
        tool_name: str,
        call_id: str,
        tool_input: dict[str, Any],
        output: Any,
        status: str,
        error_message: str | None,
        latency_ms: int,
        server_name: str | None,
    ) -> ToolCallTrace:
        return ToolCallTrace(
            call_id=call_id,
            tool_name=tool_name,
            input=tool_input,
            output=output,
            status=status,
            latency_ms=latency_ms,
            error_message=error_message,
            attempts=[],
        )

    def _request_to_state(self, request: GraphTaskRequest) -> ChartDocRAGState:
        params = request.params
        task_map = {
            "parse_document": "parse",
            "index_document": "index",
            "ask_document": "ask",
            "understand_chart": "chart_understand",
        }
        task_type = task_map.get(request.task_type, request.task_type)
        selected_skill = (
            params.get("selected_skill")
            or params.get("skill_context")
            or self.resolve_skill_context(
                task_type=request.task_type,
                preferred_skill_name=params.get("skill_name"),
            )
        )
        return {
            "request_id": request.trace_id or f"req_{uuid4().hex}",
            "task_type": task_type,
            "user_input": params.get("question")
            or params.get("file_path")
            or params.get("image_path")
            or request.task_type,
            "document_id": params.get("document_id") or params.get("doc_id"),
            "document_ids": params.get("document_ids") or [],
            "session_id": params.get("session_id"),
            "task_intent": params.get("task_intent"),
            "file_path": params.get("file_path"),
            "image_path": params.get("image_path"),
            "page_id": params.get("page_id"),
            "page_number": params.get("page_number"),
            "chart_id": params.get("chart_id"),
            "parsed_document": params.get("parsed_document"),
            "charts": params.get("charts") or [],
            "include_graph": params.get("include_graph", True),
            "include_embeddings": params.get("include_embeddings", True),
            "top_k": params.get("top_k", 10),
            "filters": params.get("filters") or {},
            "metadata": {
                **(params.get("metadata") or params.get("context") or {}),
                **(
                    {"reasoning_style": params.get("reasoning_style")}
                    if params.get("reasoning_style")
                    else {}
                ),
            },
            "retrieval_mode": params.get("retrieval_mode", "hybrid"),
            "retrieval_attempt": 0,
            "max_retrieval_attempts": params.get("max_retrieval_attempts", 1),
            "selected_skill": selected_skill,
        }

    def _state_output(self, state: ChartDocRAGState) -> Any:
        return (
            state.get("final_answer")
            or state.get("chart_result")
            or state.get("parsed_document")
            or state
        )

    def _model_from_payload(self, payload: Any) -> Any:
        return payload

    async def _run_parse_pipeline(self, state: ChartDocRAGState) -> ChartDocRAGState:
        started_at = time.perf_counter()
        try:
            parsed_document = await self.document_tools.parse_document(
                file_path=state.get("file_path") or state.get("user_input") or "",
                document_id=state.get("document_id"),
            )
            return self._merge_state(
                state,
                {
                    "parsed_document": parsed_document,
                    "document_id": parsed_document.id,
                    "document_ids": [parsed_document.id],
                    "tool_traces": [
                        self._tool_trace(
                            "parse_document",
                            "succeeded",
                            started_at=started_at,
                            metadata={"document_id": parsed_document.id},
                        )
                    ],
                },
            )
        except Exception as exc:
            return self._merge_state(
                state,
                {
                    "errors": [f"{exc.__class__.__name__}: {exc}"],
                    "tool_traces": [
                        self._tool_trace(
                            "parse_document",
                            "failed",
                            started_at=started_at,
                            error_message=str(exc),
                        )
                    ],
                },
            )

    async def _run_index_pipeline(self, state: ChartDocRAGState) -> ChartDocRAGState:
        started_at = time.perf_counter()
        parsed = state.get("parsed_document")
        if parsed is None:
            return self._merge_state(
                state,
                {
                    "errors": ["parsed_document is required for index_document"],
                    "tool_traces": [
                        self._tool_trace(
                            "index_document",
                            "failed",
                            started_at=started_at,
                            error_message="parsed_document is required for index_document",
                        )
                    ],
                },
            )

        metadata = dict(state.get("metadata") or {})
        tool_traces: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []
        graph_result = None
        graph_index = None

        try:
            if state.get("include_graph", True):
                text_blocks = [block for page in parsed.pages for block in page.text_blocks]
                graph_started = time.perf_counter()
                graph_result = await self.graph_extraction_tools.extract_from_text_blocks(
                    document_id=parsed.id,
                    text_blocks=text_blocks,
                    page_summaries=[],
                )
                tool_traces.append(
                    self._tool_trace(
                        "extract_graph_from_text_blocks",
                        graph_result.status,
                        started_at=graph_started,
                        metadata={"document_id": parsed.id},
                    )
                )
                graph_index_started = time.perf_counter()
                graph_index = await self.graph_index_service.index_graph_result(graph_result)
                metadata["graph_index"] = self._dump_model(graph_index)
                tool_traces.append(
                    self._tool_trace(
                        "index_graph_result",
                        graph_index.status,
                        started_at=graph_index_started,
                        metadata={"document_id": parsed.id},
                    )
                )
        except Exception as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
            tool_traces.append(
                self._tool_trace(
                    "graph_index_pipeline",
                    "failed",
                    started_at=started_at,
                    error_message=str(exc),
                )
            )

        try:
            if state.get("include_embeddings", True):
                text_blocks = [block for page in parsed.pages for block in page.text_blocks]
                text_result = await self.embedding_index_service.index_text_blocks(parsed.id, text_blocks)
                page_result = await self.embedding_index_service.index_pages(parsed.id, parsed.pages)
                charts = list(state.get("charts", []))
                chart_index = (
                    await self.embedding_index_service.index_charts(parsed.id, charts) if charts else None
                )
                metadata["text_embedding_index"] = self._dump_model(text_result)
                metadata["page_embedding_index"] = self._dump_model(page_result)
                metadata["chart_embedding_index"] = self._dump_model(chart_index)
                tool_traces.extend(
                    [
                        self._tool_trace(
                            "index_text_blocks",
                            text_result.status,
                            started_at=time.perf_counter(),
                            metadata={"document_id": parsed.id},
                        ),
                        self._tool_trace(
                            "index_pages",
                            page_result.status,
                            started_at=time.perf_counter(),
                            metadata={"document_id": parsed.id},
                        ),
                    ]
                )
                if chart_index is not None:
                    tool_traces.append(
                        self._tool_trace(
                            "index_charts",
                            chart_index.status,
                            started_at=time.perf_counter(),
                            metadata={"document_id": parsed.id},
                        )
                    )
        except Exception as exc:
            errors.append(f"{exc.__class__.__name__}: {exc}")
            warnings.append("Embedding indexing failed during direct tool pipeline.")
            tool_traces.append(
                self._tool_trace(
                    "embedding_index_pipeline",
                    "failed",
                    started_at=started_at,
                    error_message=str(exc),
                )
            )

        tool_traces.append(
            self._tool_trace(
                "index_document",
                "failed" if errors else "succeeded",
                started_at=started_at,
                metadata={"document_id": parsed.id},
            )
        )
        return self._merge_state(
            state,
            {
                "graph_extraction_result": graph_result,
                "metadata": metadata,
                "errors": errors,
                "warnings": warnings,
                "tool_traces": tool_traces,
            },
        )

    async def _run_chart_understand_pipeline(self, state: ChartDocRAGState) -> ChartDocRAGState:
        started_at = time.perf_counter()
        try:
            chart = await self.chart_tools.parse_chart(
                image_path=state.get("image_path") or "",
                document_id=state.get("document_id") or "unknown_document",
                page_id=state.get("page_id") or "page-1",
                page_number=state.get("page_number") or 1,
                chart_id=state.get("chart_id") or f"chart_{uuid4().hex}",
                context=state.get("metadata") or {},
            )
            chart_result = {
                "chart": chart,
                "graph_text": self.chart_tools.to_graph_text(chart),
                "metadata": {
                    "image_path": state.get("image_path"),
                    **({"skill_name": (state.get("selected_skill") or {}).get("name")} if state.get("selected_skill") else {}),
                },
            }
            return self._merge_state(
                state,
                {
                    "chart_result": chart_result,
                    "tool_traces": [
                        self._tool_trace(
                            "understand_chart",
                            "succeeded",
                            started_at=started_at,
                            metadata={"chart_id": chart.id},
                        )
                    ],
                },
            )
        except Exception as exc:
            return self._merge_state(
                state,
                {
                    "errors": [f"{exc.__class__.__name__}: {exc}"],
                    "tool_traces": [
                        self._tool_trace(
                            "understand_chart",
                            "failed",
                            started_at=started_at,
                            error_message=str(exc),
                        )
                    ],
                },
            )

    async def _run_ask_pipeline(self, state: ChartDocRAGState) -> ChartDocRAGState:
        return await self._run_answer_pipeline(state, fused=False)

    async def _run_fused_ask_pipeline(self, state: ChartDocRAGState) -> ChartDocRAGState:
        return await self._run_answer_pipeline(state, fused=True)

    async def _run_answer_pipeline(self, state: ChartDocRAGState, *, fused: bool) -> ChartDocRAGState:
        started_at = time.perf_counter()
        question = state.get("user_input") or ""
        current_state = dict(state)
        max_attempts = max(1, int(state.get("max_retrieval_attempts", 1) or 1))

        chart = None
        chart_answer = None
        chart_confidence = None
        fused_chart_hits: list[RetrievalHit] = []
        fused_chart_bundle: EvidenceBundle | None = None
        if fused:
            chart, chart_answer, chart_confidence, fused_chart_hits, fused_chart_bundle, chart_update = (
                await self._run_chart_grounding(current_state)
            )
            current_state = self._merge_state(current_state, chart_update)

        final_answer: QAResponse | None = None
        retrieval_result: HybridRetrievalResult | None = None
        evidence_bundle = EvidenceBundle()
        plan: RetrievalPlan | None = None
        validation_payload: dict[str, Any] | None = None
        warnings = list(current_state.get("warnings", []))
        errors = list(current_state.get("errors", []))
        tool_traces = list(current_state.get("tool_traces", []))
        vector_hits: list[Any] = []
        graph_hits: list[Any] = []
        graph_summary_hits: list[Any] = []

        for attempt in range(max_attempts):
            current_state["retrieval_attempt"] = attempt
            plan = await self._build_retrieval_plan(question=question, state=current_state)
            retrieval_result, retrieval_update = await self._run_retrieval_tool(
                state=current_state,
                plan=plan,
            )
            current_state = self._merge_state(current_state, retrieval_update)
            warnings = list(current_state.get("warnings", []))
            errors = list(current_state.get("errors", []))
            tool_traces = list(current_state.get("tool_traces", []))
            vector_hits = list(current_state.get("vector_hits", []))
            graph_hits = list(current_state.get("graph_hits", []))
            graph_summary_hits = list(current_state.get("graph_summary_hits", []))

            if retrieval_result is None:
                continue

            evidence_bundle = self._ensure_evidence_bundle(retrieval_result)
            if fused and fused_chart_bundle is not None:
                retrieval_result, evidence_bundle = self._merge_chart_grounding(
                    retrieval_result=retrieval_result,
                    evidence_bundle=evidence_bundle,
                    chart_hits=fused_chart_hits,
                    chart_bundle=fused_chart_bundle,
                )

            final_answer = await self._answer_with_evidence(
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                state=current_state,
                chart_answer=chart_answer,
                chart_confidence=chart_confidence,
            )
            validation = await self.validation_agent.validate(
                question=question,
                answer=final_answer,
                evidence_count=len(evidence_bundle.evidences),
                retrieval_attempt=attempt,
            )
            validation_payload = validation.model_dump(mode="json")
            current_state = self._merge_state(
                current_state,
                {
                    "validation_result": validation_payload,
                    "reasoning_summary": {
                        "validation": validation.reasoning_summary,
                        "retrieval_planner": plan.reasoning_summary,
                    },
                    "react_trace": [step.model_dump(mode="json") for step in plan.react_trace]
                    + [step.model_dump(mode="json") for step in validation.react_trace],
                    "warnings": list(validation.warnings),
                    "tool_traces": [
                        self._tool_trace(
                            "validate_answer",
                            "succeeded",
                            started_at=time.perf_counter(),
                            metadata={"decision": validation.decision},
                        )
                    ],
                },
            )
            warnings = list(current_state.get("warnings", []))
            tool_traces = list(current_state.get("tool_traces", []))
            if validation.decision != "retry_retrieval" or attempt + 1 >= max_attempts:
                break

        if final_answer is None:
            final_answer = QAResponse(
                answer="证据不足",
                question=question,
                evidence_bundle=evidence_bundle,
                retrieval_result=retrieval_result,
                confidence=0.0,
                metadata={"runtime_engine": "tool_runtime", "reason": "no_answer_generated"},
            )

        response_metadata = {
            **(current_state.get("metadata") or {}),
            "retrieval_plan": self._dump_model(plan),
            "validation_result": validation_payload,
            "fused": fused,
        }
        if chart_answer is not None:
            response_metadata["chart_answer"] = chart_answer
        if chart_confidence is not None:
            response_metadata["chart_confidence"] = chart_confidence

        return self._merge_state(
            current_state,
            {
                "final_answer": final_answer,
                "retrieval_result": retrieval_result,
                "evidence_bundle": evidence_bundle,
                "chart_answer": chart_answer,
                "chart_confidence": chart_confidence,
                "vector_hits": vector_hits,
                "graph_hits": graph_hits,
                "graph_summary_hits": graph_summary_hits,
                "metadata": response_metadata,
                "warnings": warnings,
                "errors": errors,
                "tool_traces": tool_traces
                + [
                    self._tool_trace(
                        "rag_pipeline",
                        "failed" if errors else "succeeded",
                        started_at=started_at,
                        metadata={"fused": fused},
                    )
                ],
            },
        )

    async def _build_retrieval_plan(
        self,
        *,
        question: str,
        state: ChartDocRAGState,
    ) -> RetrievalPlan:
        return await self.retrieval_planner_agent.plan(
            question=question,
            state=state,
            max_steps=3,
        )

    async def _run_retrieval_tool(
        self,
        *,
        state: ChartDocRAGState,
        plan: RetrievalPlan,
    ) -> tuple[HybridRetrievalResult | None, ChartDocRAGState]:
        started_at = time.perf_counter()
        filters = {
            **state.get("filters", {}),
            "retrieval_focus": plan.retrieval_focus,
        }
        requested_mode = filters.get("retrieval_mode") or state.get("retrieval_mode") or "hybrid"
        if "summary" in plan.modes:
            filters.setdefault("enable_graph_summary", True)
        try:
            result = await self.retrieval_tools.retrieve(
                question=plan.query or state.get("user_input") or "",
                doc_id=state.get("document_id"),
                document_ids=state.get("document_ids") or [],
                top_k=state.get("top_k", 10),
                filters=filters,
                session_id=state.get("session_id"),
                task_id=state.get("request_id"),
                memory_hints=state.get("session_memory") or {},
                skill_context=state.get("selected_skill"),
            )
            retrieval_result = result.retrieval_result
            metadata = retrieval_result.metadata or {}
            warnings = self._retrieval_warnings(metadata)
            return retrieval_result, {
                "vector_hits": metadata.get("vector_hits", retrieval_result.hits if requested_mode == "vector" else []),
                "graph_hits": metadata.get("graph_hits", retrieval_result.hits if requested_mode == "graph" else []),
                "summary_hits": metadata.get("graph_summary_hits", metadata.get("summary_hits", [])),
                "graph_summary_hits": metadata.get(
                    "graph_summary_hits",
                    retrieval_result.hits if requested_mode == "graphrag_summary" else [],
                ),
                "warnings": warnings,
                "metadata": {
                    "retrieval_focus": plan.retrieval_focus,
                    "retrieval_modes": list(plan.modes),
                    "retrieval_result": retrieval_result.model_dump(mode="json"),
                },
                "reasoning_summary": {"retrieval_planner": plan.reasoning_summary},
                "react_trace": [step.model_dump(mode="json") for step in plan.react_trace],
                "tool_traces": [
                    self._tool_trace(
                        "hybrid_retrieve",
                        "succeeded",
                        started_at=started_at,
                        metadata={
                            "requested_retrieval_mode": metadata.get(
                                "requested_retrieval_mode", requested_mode
                            ),
                            "hit_count": len(retrieval_result.hits),
                        },
                    )
                ],
            }
        except Exception as exc:
            logger.exception("Direct retrieval tool failed")
            return None, {
                "errors": [f"{exc.__class__.__name__}: {exc}"],
                "warnings": ["Retrieval tool failed during direct RAG pipeline."],
                "tool_traces": [
                    self._tool_trace(
                        "hybrid_retrieve",
                        "failed",
                        started_at=started_at,
                        error_message=str(exc),
                    )
                ],
            }

    def _retrieval_warnings(self, metadata: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        failed_sources = metadata.get("failed_sources") or []
        for source in failed_sources:
            warnings.append(f"{source} retrieval unavailable; continuing with remaining evidence sources.")
        return warnings

    def _ensure_evidence_bundle(self, retrieval_result: HybridRetrievalResult) -> EvidenceBundle:
        bundle = retrieval_result.evidence_bundle
        if bundle.evidences:
            return bundle
        return build_evidence_bundle(retrieval_result.hits)

    async def _answer_with_evidence(
        self,
        *,
        question: str,
        evidence_bundle: EvidenceBundle,
        retrieval_result: HybridRetrievalResult | None,
        state: ChartDocRAGState,
        chart_answer: str | None = None,
        chart_confidence: float | None = None,
    ) -> QAResponse:
        answer_fn = getattr(self.answer_tools, "answer_with_evidence", None)
        if not callable(answer_fn):
            answer_fn = getattr(self.answer_tools, "answer")
        task_context = {
            "task_intent": state.get("task_intent"),
            "document_id": state.get("document_id"),
            "document_ids": state.get("document_ids") or [],
            **({"chart_answer": chart_answer} if chart_answer is not None else {}),
            **({"chart_confidence": chart_confidence} if chart_confidence is not None else {}),
        }
        return await answer_fn(
            question=question,
            evidence_bundle=evidence_bundle,
            retrieval_result=retrieval_result,
            metadata=state.get("metadata") or {},
            session_context=state.get("session_memory") or {},
            task_context=task_context,
            preference_context={"reasoning_style": (state.get("metadata") or {}).get("reasoning_style")},
            memory_hints=state.get("session_memory") or {},
            skill_context=state.get("selected_skill") or {},
        )

    async def _run_chart_grounding(
        self,
        state: ChartDocRAGState,
    ) -> tuple[
        ChartSchema | None,
        str | None,
        float | None,
        list[RetrievalHit],
        EvidenceBundle | None,
        ChartDocRAGState,
    ]:
        started_at = time.perf_counter()
        try:
            chart = await self.chart_tools.parse_chart(
                image_path=state.get("image_path") or "",
                document_id=state.get("document_id") or "unknown_document",
                page_id=state.get("page_id") or "page-1",
                page_number=state.get("page_number") or 1,
                chart_id=state.get("chart_id") or f"chart_{uuid4().hex}",
                context=self._chart_question_context(state),
            )
            visible_text = ""
            extract_visible_text = getattr(self.chart_tools, "extract_visible_text", None)
            if callable(extract_visible_text):
                visible_text = await extract_visible_text(
                    image_path=state.get("image_path") or "",
                    context=self._chart_question_context(state),
                    chart=chart,
                )
            ask_chart = getattr(self.chart_tools, "ask_chart", None)
            chart_answer = (
                await ask_chart(
                    image_path=state.get("image_path") or "",
                    question=state.get("user_input") or "",
                    context=self._chart_question_context(state),
                    history=self.session_memory.chart_history(
                        state.get("session_id"),
                        image_path=state.get("image_path"),
                    ),
                )
                if callable(ask_chart)
                else None
            )
            chart_confidence = self._estimate_chart_confidence(chart_answer or visible_text or chart.summary or "")
            snippet = self._compose_chart_grounding_snippet(
                chart=chart,
                chart_answer=chart_answer,
                visible_text=visible_text,
                metadata=state.get("metadata") or {},
            )
            chart_evidence = Evidence(
                id=f"ev:chart:{chart.id}",
                document_id=chart.document_id,
                page_id=chart.page_id,
                page_number=chart.page_number,
                source_type="chart",
                source_id=chart.id,
                snippet=snippet,
                score=chart_confidence,
                metadata={
                    "title": chart.title,
                    "caption": chart.caption,
                    "image_path": state.get("image_path"),
                    "chart_type": chart.chart_type,
                    **self._chart_figure_metadata(state.get("metadata") or {}),
                },
            )
            page_evidence = Evidence(
                id=f"ev:page_image:{chart.page_id}",
                document_id=chart.document_id,
                page_id=chart.page_id,
                page_number=chart.page_number,
                source_type="page_image",
                source_id=chart.page_id,
                snippet=(chart_answer or snippet)[:800] or chart.summary,
                score=chart_confidence,
                metadata={"image_path": state.get("image_path"), "chart_id": chart.id},
            )
            chart_hit = RetrievalHit(
                id=f"hit:chart:{chart.id}",
                source_type="chart",
                source_id=chart.id,
                document_id=chart.document_id,
                content=snippet,
                merged_score=chart_confidence,
                metadata={
                    "page_id": chart.page_id,
                    "page_number": chart.page_number,
                    "image_path": state.get("image_path"),
                    "title": chart.title,
                    "chart_type": chart.chart_type,
                    **self._chart_figure_metadata(state.get("metadata") or {}),
                },
            )
            page_hit = RetrievalHit(
                id=f"hit:page_image:{chart.page_id}",
                source_type="page_image",
                source_id=chart.page_id,
                document_id=chart.document_id,
                content=(chart_answer or snippet)[:800] or chart.summary,
                merged_score=chart_confidence,
                metadata={
                    "page_id": chart.page_id,
                    "page_number": chart.page_number,
                    "image_path": state.get("image_path"),
                },
            )
            chart_update = {
                "chart_result": {
                    "chart": chart,
                    "graph_text": self.chart_tools.to_graph_text(chart),
                    "metadata": {"image_path": state.get("image_path")},
                },
                "chart_answer": chart_answer,
                "chart_confidence": chart_confidence,
                "tool_traces": [
                    self._tool_trace(
                        "chart_grounding",
                        "succeeded",
                        started_at=started_at,
                        metadata={"chart_id": chart.id},
                    )
                ],
            }
            return (
                chart,
                chart_answer,
                chart_confidence,
                [chart_hit, page_hit],
                EvidenceBundle(
                    evidences=[chart_evidence, page_evidence],
                    summary=chart.summary or snippet[:400] or None,
                metadata={"chart_id": chart.id, "source": "chart_grounding"},
                ),
                chart_update,
            )
        except Exception as exc:
            logger.exception("Chart grounding failed in fused ask pipeline")
            return (
                None,
                None,
                None,
                [],
                None,
                {
                    "warnings": ["Chart grounding failed; continuing with retrieval evidence only."],
                    "tool_traces": [
                        self._tool_trace(
                            "chart_grounding",
                            "failed",
                            started_at=started_at,
                            error_message=str(exc),
                        )
                    ],
                },
            )

    def _merge_chart_grounding(
        self,
        *,
        retrieval_result: HybridRetrievalResult,
        evidence_bundle: EvidenceBundle,
        chart_hits: list[RetrievalHit],
        chart_bundle: EvidenceBundle,
    ) -> tuple[HybridRetrievalResult, EvidenceBundle]:
        merged_evidences = list(chart_bundle.evidences)
        seen_ids = {evidence.id for evidence in merged_evidences}
        for evidence in evidence_bundle.evidences:
            if evidence.id not in seen_ids:
                merged_evidences.append(evidence)
        fused_bundle = EvidenceBundle(
            evidences=merged_evidences,
            summary=chart_bundle.summary or evidence_bundle.summary,
            metadata={**evidence_bundle.metadata, **chart_bundle.metadata, "fused": True},
        )
        fused_result = retrieval_result.model_copy(
            update={
                "hits": [*chart_hits, *retrieval_result.hits],
                "evidence_bundle": fused_bundle,
                "metadata": {**retrieval_result.metadata, "fused_chart_grounding": True},
            }
        )
        return fused_result, fused_bundle

    def _chart_question_context(self, state: ChartDocRAGState) -> dict[str, Any]:
        metadata = state.get("metadata") or {}
        allowed_metadata_keys = {
            "research_task_id",
            "research_topic",
            "qa_route",
            "qa_scope_mode",
            "selected_paper_ids",
            "selected_document_ids",
            "selected_paper_titles",
            "selection_summary",
            "visual_anchor",
            "visual_anchor_figure",
            "figure_context",
            "reasoning_style",
            "user_intent",
        }
        return {
            "document_id": state.get("document_id"),
            "page_id": state.get("page_id"),
            "page_number": state.get("page_number"),
            "chart_id": state.get("chart_id"),
            **{
                key: self._compact_chart_context_value(value)
                for key, value in metadata.items()
                if key in allowed_metadata_keys
            },
        }

    def _compose_chart_grounding_snippet(
        self,
        *,
        chart: ChartSchema,
        chart_answer: str | None,
        visible_text: str | None,
        metadata: dict[str, Any],
    ) -> str:
        parts: list[str] = []
        figure_context = self._resolve_figure_context(metadata)
        figure_overview = self._format_figure_context(figure_context)
        if figure_overview:
            parts.append(figure_overview)
        chart_overview = explain_chart(chart).strip()
        if chart_overview:
            parts.append(chart_overview)
        if chart_answer:
            parts.append(f"Visual answer: {chart_answer.strip()}")
        structured_visible = visible_chart_text(chart).strip()
        if structured_visible:
            parts.append(f"Structured visible text:\n{structured_visible[:800]}")
        cleaned_visible_text = (visible_text or "").strip()
        if cleaned_visible_text:
            parts.append(f"Visible text: {cleaned_visible_text[:800]}")
        composed = "\n".join(part for part in parts if part).strip()
        if composed:
            return composed[:1600]
        return cleaned_visible_text[:1600] or (chart.summary or "")[:1600]

    def _resolve_figure_context(self, metadata: dict[str, Any]) -> dict[str, Any]:
        figure_context = metadata.get("figure_context")
        if isinstance(figure_context, dict) and figure_context:
            return figure_context
        visual_anchor_figure = metadata.get("visual_anchor_figure")
        if isinstance(visual_anchor_figure, dict) and visual_anchor_figure:
            return visual_anchor_figure
        return {}

    def _format_figure_context(self, figure_context: dict[str, Any]) -> str:
        if not figure_context:
            return ""
        parts: list[str] = []
        title = str(figure_context.get("title") or "").strip()
        caption = str(figure_context.get("caption") or "").strip()
        source = str(figure_context.get("source") or "").strip()
        figure_id = str(figure_context.get("figure_id") or "").strip()
        if title:
            parts.append(f"Figure title: {title}")
        if caption:
            parts.append(f"Figure caption: {caption}")
        if source:
            parts.append(f"Figure source: {source}")
        if figure_id:
            parts.append(f"Figure id: {figure_id}")
        return "\n".join(parts)

    def _chart_figure_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        figure_context = self._resolve_figure_context(metadata)
        allowed_keys = {"paper_id", "figure_id", "chart_id", "page_id", "page_number", "title", "caption", "source"}
        return {
            f"figure_{key}": value
            for key, value in figure_context.items()
            if key in allowed_keys and value is not None
        }

    def _compact_chart_context_value(
        self,
        value: Any,
        *,
        max_depth: int = 2,
        max_items: int = 6,
        max_text: int = 240,
    ) -> Any:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            normalized = value.strip()
            if len(normalized) <= max_text:
                return normalized
            return f"{normalized[:max_text]}..."
        if max_depth <= 0:
            if isinstance(value, dict):
                return f"<dict:{len(value)}>"
            if isinstance(value, list):
                return f"<list:{len(value)}>"
            return str(value)[:max_text]
        if isinstance(value, list):
            compacted = [
                self._compact_chart_context_value(
                    item,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_text=max_text,
                )
                for item in value[:max_items]
            ]
            if len(value) > max_items:
                compacted.append(f"...(+{len(value) - max_items} more)")
            return compacted
        if isinstance(value, dict):
            compacted_dict: dict[str, Any] = {}
            for key, nested_value in list(value.items())[:max_items]:
                compacted_dict[str(key)] = self._compact_chart_context_value(
                    nested_value,
                    max_depth=max_depth - 1,
                    max_items=max_items,
                    max_text=max_text,
                )
            if len(value) > max_items:
                compacted_dict["_truncated"] = f"+{len(value) - max_items} more"
            return compacted_dict
        return str(value)[:max_text]

    def _estimate_chart_confidence(self, answer: str) -> float:
        lowered = answer.lower()
        if any(token in lowered for token in ["无法", "看不", "not readable", "cannot", "不确定"]):
            return 0.55
        if len(answer.strip()) < 20:
            return 0.45
        return 0.78

    def _tool_trace(
        self,
        tool_name: str,
        status: str,
        *,
        started_at: float,
        metadata: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> dict[str, Any]:
        trace = {
            "tool_name": tool_name,
            "status": status,
            "latency_ms": round((time.perf_counter() - started_at) * 1000, 2),
        }
        if metadata:
            trace["metadata"] = metadata
        if error_message:
            trace["error_message"] = error_message
        return trace

    def _merge_state(self, left: ChartDocRAGState, right: ChartDocRAGState) -> ChartDocRAGState:
        append_keys = {
            "vector_hits",
            "graph_hits",
            "summary_hits",
            "graph_summary_hits",
            "warnings",
            "react_trace",
            "messages",
            "tool_traces",
            "errors",
        }
        merge_keys = {"reasoning_summary", "metadata"}
        merged = dict(left)
        for key, value in right.items():
            if key in append_keys:
                merged[key] = [*(merged.get(key) or []), *(value or [])]
                continue
            if key in merge_keys:
                merged[key] = {**(merged.get(key) or {}), **(value or {})}
                continue
            merged[key] = value
        return merged

    def _dump_model(self, value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        return value


GraphRuntime = RagRuntime
