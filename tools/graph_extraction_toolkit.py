import asyncio
import logging
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from adapters.llm.base import (
    BaseLLMAdapter,
    LLMAdapterError,
    format_llm_error,
    is_expected_provider_error,
)
from adapters.local_runtime import LocalLLMAdapter
from chains.graph_extraction_chain import GraphExtractionChain
from domain.schemas.chart import ChartSchema
from domain.schemas.document import TextBlock
from domain.schemas.graph import GraphEdge, GraphExtractionResult, GraphNode, GraphTriple

logger = logging.getLogger(__name__)


class GraphExtractionAgentError(RuntimeError):
    """Raised when graph extraction fails."""


class PageSummaryInput(BaseModel):
    page_id: str
    page_number: int = Field(..., ge=1)
    summary: str


class GraphFromTextBlocksInput(BaseModel):
    document_id: str
    text_blocks: list[TextBlock] = Field(default_factory=list)
    page_summaries: list[PageSummaryInput] = Field(default_factory=list)


class GraphFromChartInput(BaseModel):
    chart: ChartSchema
    chart_summary: str | None = None


def merge_graph_candidates_run(
    *,
    document_id: str,
    candidates: list[GraphExtractionResult],
) -> GraphExtractionResult:
    nodes: dict[str, GraphNode] = {}
    edges: dict[str, GraphEdge] = {}
    triples: dict[str, GraphTriple] = {}
    degraded_count = 0
    failed_count = 0

    for candidate in candidates:
        if candidate.status != "succeeded":
            degraded_count += 1
        if candidate.status == "failed":
            failed_count += 1
        for node in candidate.nodes:
            nodes.setdefault(node.id, node)
        for edge in candidate.edges:
            edges.setdefault(edge.id, edge)
        for triple in candidate.triples:
            nodes.setdefault(triple.subject.id, triple.subject)
            nodes.setdefault(triple.object.id, triple.object)
            edges.setdefault(triple.predicate.id, triple.predicate)
            triples.setdefault(f"{triple.subject.id}:{triple.predicate.type}:{triple.object.id}", triple)

    valid_edges = {
        edge_id: edge
        for edge_id, edge in edges.items()
        if edge.source_node_id in nodes and edge.target_node_id in nodes
    }
    valid_triples = {
        key: triple
        for key, triple in triples.items()
        if triple.subject.id in nodes and triple.object.id in nodes and triple.predicate.id in valid_edges
    }

    status = "succeeded"
    if degraded_count and (nodes or valid_edges or valid_triples):
        status = "partial"
    elif degraded_count and not (nodes or valid_edges or valid_triples):
        status = "failed"

    return GraphExtractionResult(
        document_id=document_id,
        nodes=list(nodes.values()),
        edges=list(valid_edges.values()),
        triples=list(valid_triples.values()),
        status=status,
        error_message=None if status != "failed" else "All graph extraction candidates failed",
        metadata={
            "candidate_count": len(candidates),
            "degraded_candidate_count": degraded_count,
            "failed_candidate_count": failed_count,
            "dropped_edge_count": len(edges) - len(valid_edges),
            "dropped_triple_count": len(triples) - len(valid_triples),
        },
    )


class GraphExtractionAgent:
    def __init__(
        self,
        llm_adapter: BaseLLMAdapter | None = None,
        prompt_path: str | Path = "prompts/graph/extract_triples.txt",
        text_graph_timeout_seconds: float = 12.0,
        text_graph_chunk_size: int = 48,
        text_graph_chunk_chars: int = 24000,
        max_text_graph_chunks: int = 6,
    ) -> None:
        self.llm_adapter = llm_adapter or LocalLLMAdapter()
        self.prompt_path = Path(prompt_path)
        self.text_graph_timeout_seconds = max(text_graph_timeout_seconds, 0.0)
        self.text_graph_chunk_size = max(int(text_graph_chunk_size), 1)
        self.text_graph_chunk_chars = max(int(text_graph_chunk_chars), 1)
        self.max_text_graph_chunks = max(int(max_text_graph_chunks), 1)
        self.chain = GraphExtractionChain(llm=self.llm_adapter, prompt_path=self.prompt_path)
        self.extract_from_text_blocks_tool = StructuredTool.from_function(
            coroutine=self.extract_from_text_blocks,
            name="extract_graph_from_text_blocks",
            description="Extract graph triples from document text blocks.",
            args_schema=GraphFromTextBlocksInput,
        )
        self.extract_from_chart_tool = StructuredTool.from_function(
            coroutine=self.extract_from_chart,
            name="extract_graph_from_chart",
            description="Extract graph triples from structured chart content.",
            args_schema=GraphFromChartInput,
        )
        self.merge_graph_candidates_chain = RunnableLambda(
            lambda payload: merge_graph_candidates_run(document_id=payload["document_id"], candidates=payload["candidates"])
        )

    async def extract_from_text_blocks(
        self,
        document_id: str,
        text_blocks: list[TextBlock],
        page_summaries: list[PageSummaryInput] | None = None,
    ) -> GraphExtractionResult:
        resolved_page_summaries = page_summaries or []
        if not text_blocks and not resolved_page_summaries:
            return GraphExtractionResult(
                document_id=document_id,
                status="succeeded",
                metadata={"source": "text_blocks", "reason": "empty_input"},
            )
        chunk_inputs = self._chunk_text_block_inputs(
            text_blocks=text_blocks,
            page_summaries=resolved_page_summaries,
        )
        original_chunk_count = len(chunk_inputs)
        if len(chunk_inputs) > self.max_text_graph_chunks:
            logger.info(
                "Graph extraction chunk cap applied",
                extra={
                    "document_id": document_id,
                    "original_chunk_count": original_chunk_count,
                    "max_text_graph_chunks": self.max_text_graph_chunks,
                },
            )
            chunk_inputs = chunk_inputs[: self.max_text_graph_chunks]
        chunk_results: list[GraphExtractionResult] = []
        for chunk_index, (chunk_blocks, chunk_summaries) in enumerate(chunk_inputs, start=1):
            chunk_results.append(
                await self._extract_text_block_candidate(
                    document_id=document_id,
                    text_blocks=chunk_blocks,
                    page_summaries=chunk_summaries,
                    chunk_index=chunk_index,
                    chunk_count=len(chunk_inputs),
                )
            )
        return self._finalize_text_block_results(
            document_id=document_id,
            chunk_results=chunk_results,
            text_block_count=len(text_blocks),
            page_summary_count=len(resolved_page_summaries),
            original_chunk_count=original_chunk_count,
        )

    async def _extract_text_block_candidate(
        self,
        *,
        document_id: str,
        text_blocks: list[TextBlock],
        page_summaries: list[PageSummaryInput],
        chunk_index: int,
        chunk_count: int,
    ) -> GraphExtractionResult:
        chunk_metadata = self._chunk_metadata(
            text_blocks=text_blocks,
            page_summaries=page_summaries,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        )
        try:
            summary_payload = [summary.model_dump(mode="json") for summary in page_summaries]
            chain_call = self.chain.ainvoke_from_text_blocks(
                document_id=document_id,
                text_blocks=text_blocks,
                page_summaries=summary_payload,
            )
            result = (
                await asyncio.wait_for(chain_call, timeout=self.text_graph_timeout_seconds)
                if self.text_graph_timeout_seconds > 0
                else await chain_call
            )
            return self._normalize_result(document_id, result, source="text_blocks", **chunk_metadata)
        except (asyncio.TimeoutError, LLMAdapterError, OSError, ValueError) as exc:
            should_fallback = self._should_fallback(exc)
            self._log_failure(
                "text blocks",
                exc,
                document_id=document_id,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
            )
            if should_fallback:
                fallback_result = await self._fallback_extract_from_text_blocks(
                    document_id=document_id,
                    text_blocks=text_blocks,
                    page_summaries=page_summaries,
                )
                if fallback_result is not None:
                    return fallback_result.model_copy(
                        update={
                            "metadata": {
                                **fallback_result.metadata,
                                "source": "text_blocks",
                                "fallback": True,
                                "fallback_mode": "local_extract",
                                "expected_provider_error": should_fallback,
                                "error_detail": self._error_detail(exc),
                                **chunk_metadata,
                            }
                        }
                    )
            return GraphExtractionResult(
                document_id=document_id,
                status="failed",
                error_message=str(exc),
                metadata={
                    "source": "text_blocks",
                    "expected_provider_error": should_fallback,
                    "error_detail": self._error_detail(exc),
                    **chunk_metadata,
                },
            )

    async def extract_from_chart(
        self,
        chart: ChartSchema,
        chart_summary: str | None = None,
    ) -> GraphExtractionResult:
        summary = chart_summary or chart.summary or ""
        if not summary and not chart.series:
            return GraphExtractionResult(
                document_id=chart.document_id,
                status="succeeded",
                metadata={"source": "chart", "chart_id": chart.id, "reason": "empty_input"},
            )
        try:
            result = await self.chain.ainvoke_from_chart(chart=chart, chart_summary=summary)
            return self._normalize_result(chart.document_id, result, source="chart", chart_id=chart.id)
        except (LLMAdapterError, OSError, ValueError) as exc:
            self._log_failure("chart", exc, chart_id=chart.id, document_id=chart.document_id)
            return GraphExtractionResult(
                document_id=chart.document_id,
                status="failed",
                error_message=str(exc),
                metadata={
                    "source": "chart",
                    "chart_id": chart.id,
                    "expected_provider_error": is_expected_provider_error(exc),
                    "error_detail": self._error_detail(exc),
                },
            )

    def merge_graph_candidates(
        self,
        document_id: str,
        candidates: list[GraphExtractionResult],
    ) -> GraphExtractionResult:
        return merge_graph_candidates_run(document_id=document_id, candidates=candidates)

    def prepare_community_input(self, graph_result: GraphExtractionResult) -> GraphExtractionResult:
        nodes = {node.id: node for node in graph_result.nodes}
        edges = {edge.id: edge for edge in graph_result.edges}
        triples = {}
        for triple in graph_result.triples:
            nodes.setdefault(triple.subject.id, triple.subject)
            nodes.setdefault(triple.object.id, triple.object)
            edges.setdefault(triple.predicate.id, triple.predicate)
            triples[self._triple_key(triple)] = triple
        return GraphExtractionResult(
            document_id=graph_result.document_id,
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            triples=list(triples.values()),
            status=graph_result.status,
            error_message=graph_result.error_message,
            metadata={**graph_result.metadata, "community_ready": True},
        )

    def _normalize_result(
        self,
        document_id: str,
        result: GraphExtractionResult,
        source: str,
        **metadata: Any,
    ) -> GraphExtractionResult:
        nodes = {
            node.id: self._hydrate_node_source_reference(node=node, document_id=document_id, source=source)
            for node in result.nodes
        }
        edges = {
            edge.id: self._hydrate_edge_source_reference(
                edge=edge,
                document_id=document_id,
                source=source,
                nodes=nodes,
            )
            for edge in result.edges
            if edge.source_node_id in nodes and edge.target_node_id in nodes
        }
        triples: list[GraphTriple] = []
        for triple in result.triples:
            if triple.subject.id not in nodes or triple.object.id not in nodes or triple.predicate.id not in edges:
                continue
            triples.append(
                GraphTriple(
                    subject=nodes[triple.subject.id],
                    predicate=edges[triple.predicate.id],
                    object=nodes[triple.object.id],
                )
            )
        status = result.status
        if result.status == "succeeded" and (len(edges) != len(result.edges) or len(triples) != len(result.triples)):
            status = "partial"
        return GraphExtractionResult(
            document_id=document_id,
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            triples=triples,
            status=status,
            error_message=result.error_message,
            metadata={**result.metadata, "source": source, **metadata},
        )

    def _hydrate_node_source_reference(
        self,
        *,
        node: GraphNode,
        document_id: str,
        source: str,
    ) -> GraphNode:
        source_reference = node.source_reference
        if not source_reference.document_id:
            source_reference = source_reference.model_copy(update={"document_id": document_id})
        if not source_reference.source_id:
            source_reference = source_reference.model_copy(update={"source_id": node.id})
        if source_reference.source_type == "graph_node" and source == "text_blocks":
            source_reference = source_reference.model_copy(update={"source_type": "text_block"})
        if source_reference.snippet is None:
            snippet = str(node.properties.get("name") or node.properties.get("title") or node.label or node.id)
            source_reference = source_reference.model_copy(update={"snippet": snippet[:500]})
        return node.model_copy(update={"source_reference": source_reference})

    def _hydrate_edge_source_reference(
        self,
        *,
        edge: GraphEdge,
        document_id: str,
        source: str,
        nodes: dict[str, GraphNode],
    ) -> GraphEdge:
        source_reference = edge.source_reference
        if not source_reference.document_id:
            source_reference = source_reference.model_copy(update={"document_id": document_id})
        if not source_reference.source_id:
            source_reference = source_reference.model_copy(update={"source_id": edge.id})
        if source_reference.source_type == "graph_edge" and source == "text_blocks":
            source_reference = source_reference.model_copy(update={"source_type": "text_block"})
        if source_reference.snippet is None:
            source_label = nodes.get(edge.source_node_id).label if edge.source_node_id in nodes else edge.source_node_id
            target_label = nodes.get(edge.target_node_id).label if edge.target_node_id in nodes else edge.target_node_id
            source_reference = source_reference.model_copy(
                update={"snippet": f"{source_label} -[{edge.type}]-> {target_label}"[:500]}
            )
        return edge.model_copy(update={"source_reference": source_reference})

    def _triple_key(self, triple: GraphTriple) -> str:
        return f"{triple.subject.id}:{triple.predicate.id}:{triple.object.id}"

    def _chunk_text_block_inputs(
        self,
        *,
        text_blocks: list[TextBlock],
        page_summaries: list[PageSummaryInput],
    ) -> list[tuple[list[TextBlock], list[PageSummaryInput]]]:
        if not text_blocks:
            return self._chunk_summary_only_inputs(page_summaries)

        chunks: list[tuple[list[TextBlock], list[PageSummaryInput]]] = []
        current_blocks: list[TextBlock] = []
        current_chars = 0

        def flush_current_blocks() -> None:
            nonlocal current_blocks, current_chars
            if not current_blocks:
                return
            page_ids = {block.page_id for block in current_blocks}
            chunks.append((current_blocks, self._select_page_summaries(page_summaries, page_ids)))
            current_blocks = []
            current_chars = 0

        for block in text_blocks:
            block_chars = self._content_length(block.text)
            should_flush = bool(current_blocks) and (
                len(current_blocks) >= self.text_graph_chunk_size
                or current_chars + block_chars > self.text_graph_chunk_chars
            )
            if should_flush:
                flush_current_blocks()
            current_blocks.append(block)
            current_chars += block_chars
        flush_current_blocks()

        covered_page_ids = {summary.page_id for _, summaries in chunks for summary in summaries}
        orphan_summaries = [summary for summary in page_summaries if summary.page_id not in covered_page_ids]
        if orphan_summaries:
            chunks.extend(self._chunk_summary_only_inputs(orphan_summaries))
        return chunks

    def _chunk_summary_only_inputs(
        self,
        page_summaries: list[PageSummaryInput],
    ) -> list[tuple[list[TextBlock], list[PageSummaryInput]]]:
        chunks: list[tuple[list[TextBlock], list[PageSummaryInput]]] = []
        current_summaries: list[PageSummaryInput] = []
        current_chars = 0

        def flush_current_summaries() -> None:
            nonlocal current_summaries, current_chars
            if not current_summaries:
                return
            chunks.append(([], current_summaries))
            current_summaries = []
            current_chars = 0

        for summary in page_summaries:
            summary_chars = self._content_length(summary.summary)
            should_flush = bool(current_summaries) and (
                len(current_summaries) >= self.text_graph_chunk_size
                or current_chars + summary_chars > self.text_graph_chunk_chars
            )
            if should_flush:
                flush_current_summaries()
            current_summaries.append(summary)
            current_chars += summary_chars
        flush_current_summaries()
        return chunks

    def _select_page_summaries(
        self,
        page_summaries: list[PageSummaryInput],
        page_ids: set[str],
    ) -> list[PageSummaryInput]:
        return [summary for summary in page_summaries if summary.page_id in page_ids]

    def _content_length(self, value: str) -> int:
        stripped = value.strip()
        return max(len(stripped or value), 1)

    def _chunk_metadata(
        self,
        *,
        text_blocks: list[TextBlock],
        page_summaries: list[PageSummaryInput],
        chunk_index: int,
        chunk_count: int,
    ) -> dict[str, Any]:
        if chunk_count <= 1:
            return {}
        return {
            "chunk_index": chunk_index,
            "chunk_count": chunk_count,
            "chunk_text_block_count": len(text_blocks),
            "chunk_page_summary_count": len(page_summaries),
        }

    def _finalize_text_block_results(
        self,
        *,
        document_id: str,
        chunk_results: list[GraphExtractionResult],
        text_block_count: int,
        page_summary_count: int,
        original_chunk_count: int | None = None,
    ) -> GraphExtractionResult:
        chunk_count = len(chunk_results)
        base_result = chunk_results[0] if chunk_count == 1 else self.merge_graph_candidates(document_id, chunk_results)
        fallback_chunk_count = sum(1 for result in chunk_results if result.metadata.get("fallback"))
        failed_chunk_count = sum(1 for result in chunk_results if result.status == "failed")
        partial_chunk_count = sum(1 for result in chunk_results if result.status == "partial")
        degraded_chunk_indexes = [
            index
            for index, result in enumerate(chunk_results, start=1)
            if result.status != "succeeded" or result.metadata.get("fallback")
        ]
        metadata = {
            **base_result.metadata,
            "source": "text_blocks",
            "chunked": chunk_count > 1,
            "chunk_count": chunk_count,
            "original_chunk_count": original_chunk_count or chunk_count,
            "chunk_cap_applied": bool(original_chunk_count and original_chunk_count > chunk_count),
            "text_block_count": text_block_count,
            "page_summary_count": page_summary_count,
            "fallback": fallback_chunk_count > 0,
            "fallback_chunk_count": fallback_chunk_count,
            "failed_chunk_count": failed_chunk_count,
            "partial_chunk_count": partial_chunk_count,
            "expected_provider_error": any(
                bool(result.metadata.get("expected_provider_error")) for result in chunk_results
            ),
        }
        if fallback_chunk_count:
            metadata["fallback_mode"] = "local_extract"
        if degraded_chunk_indexes:
            metadata["degraded_chunk_indexes"] = degraded_chunk_indexes
        return base_result.model_copy(update={"metadata": metadata})

    async def _fallback_extract_from_text_blocks(
        self,
        *,
        document_id: str,
        text_blocks: list[TextBlock],
        page_summaries: list[PageSummaryInput],
    ) -> GraphExtractionResult | None:
        try:
            local_adapter = LocalLLMAdapter()
            payload_blocks = [block.model_dump(mode="json") for block in text_blocks]
            for summary in page_summaries:
                payload_blocks.append(
                    {
                        "id": f"summary_{summary.page_id}",
                        "document_id": document_id,
                        "page_id": summary.page_id,
                        "page_number": summary.page_number,
                        "text": summary.summary,
                        "block_type": "paragraph",
                        "metadata": {"source": "page_summary"},
                    }
                )
            result = await local_adapter.extract_graph_triples(
                prompt="Extract graph triples from document text blocks.",
                input_data={"document_id": document_id, "text_blocks": payload_blocks},
                response_model=GraphExtractionResult,
            )
            return self._normalize_result(document_id, result, source="text_blocks", fallback_mode="local_extract")
        except Exception:
            logger.exception("Failed to generate local graph extraction fallback", extra={"document_id": document_id})
            return None

    def _log_failure(self, source: str, exc: Exception, **extra: Any) -> None:
        if self._should_fallback(exc):
            logger.warning(
                "Graph extraction degraded for %s: %s",
                source,
                self._error_detail(exc),
                extra=extra,
            )
            return
        logger.exception("Failed to extract graph triples from %s", source, extra=extra)

    def _error_detail(self, exc: Exception) -> str:
        if isinstance(exc, asyncio.TimeoutError):
            return f"TimeoutError: graph extraction timed out after {self.text_graph_timeout_seconds:.1f}s"
        cause = exc.__cause__
        if isinstance(cause, Exception) and is_expected_provider_error(cause):
            return format_llm_error(cause)
        return format_llm_error(exc)

    def _should_fallback(self, exc: Exception) -> bool:
        return isinstance(exc, asyncio.TimeoutError) or is_expected_provider_error(exc)


GraphExtractionTools = GraphExtractionAgent
GraphExtractionToolsError = GraphExtractionAgentError
