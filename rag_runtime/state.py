from __future__ import annotations

from typing import Any, Literal, TypeVar, TypedDict

from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import ParsedDocument
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.graph import GraphExtractionResult
from domain.schemas.retrieval import RetrievalHit

TaskType = Literal[
    "parse",
    "index",
    "ask",
    "chart_understand",
    "graph_aware_answer",
    "function_call",
]

RetrievalMode = Literal["vector", "graph", "hybrid", "graphrag_summary"]

NextAction = Literal[
    "document_understanding",
    "chart_vision",
    "graph_extraction",
    "embedding_index",
    "graph_index",
    "retrieval_planner",
    "retrieve_vector",
    "retrieve_graph",
    "retrieve_graph_summary",
    "merge_evidence",
    "answer",
    "validation",
    "finish",
]


T = TypeVar("T")


def append_list(left: list[T], right: list[T] | None) -> list[T]:
    if not right:
        return left
    return [*left, *right]


def merge_dict(left: dict[str, Any], right: dict[str, Any] | None) -> dict[str, Any]:
    if not right:
        return left
    return {**left, **right}


class ToolTrace(TypedDict, total=False):
    trace_id: str
    node_name: str
    tool_name: str
    status: Literal["started", "succeeded", "failed", "skipped"]
    input: dict[str, Any]
    output: Any
    error_message: str | None
    latency_ms: float | None
    metadata: dict[str, Any]


class ChartResult(TypedDict, total=False):
    chart: ChartSchema
    graph_text: str
    metadata: dict[str, Any]


class FinalAnswer(TypedDict, total=False):
    answer: str
    question: str
    evidence_bundle: EvidenceBundle
    confidence: float | None
    metadata: dict[str, Any]


class ChartDocRAGState(TypedDict, total=False):
    # Request envelope and routing
    request_id: str
    task_type: TaskType
    user_input: str
    next_action: NextAction | None
    next_node: str | None
    retrieval_attempt: int
    max_retrieval_attempts: int

    # Optional identity and checkpoint keys
    thread_id: str
    session_id: str | None
    user_id: str | None
    task_intent: str | None

    # Document and multimodal inputs
    document_id: str | None
    document_ids: list[str]
    file_path: str | None
    image_path: str | None
    page_id: str | None
    page_number: int | None
    chart_id: str | None

    # Understanding outputs
    parsed_document: ParsedDocument | None
    charts: list[ChartSchema]
    chart_result: ChartResult | None
    chart_answer: str | None
    chart_confidence: float | None
    graph_extraction_result: GraphExtractionResult | None
    retrieval_plan: dict[str, Any] | None
    validation_result: dict[str, Any] | None
    route_plan: dict[str, Any] | None

    # Retrieval outputs
    vector_hits: list[RetrievalHit]
    graph_hits: list[RetrievalHit]
    summary_hits: list[RetrievalHit]
    graph_summary_hits: list[RetrievalHit]
    evidence_bundle: EvidenceBundle | None
    session_memory: dict[str, Any] | None

    # Answering and verification
    final_answer: QAResponse | FinalAnswer | None
    confidence: float | None
    warnings: list[str]
    reasoning_summary: dict[str, str]
    react_trace: list[dict[str, Any]]

    # Tool/runtime interaction context
    messages: list[Any]
    tool_traces: list[ToolTrace]

    # Shared execution context
    retrieval_mode: RetrievalMode
    include_graph: bool
    include_embeddings: bool
    top_k: int
    filters: dict[str, Any]
    errors: list[str]
    metadata: dict[str, Any]


class GraphInput(TypedDict, total=False):
    request_id: str
    task_type: TaskType
    user_input: str
    document_id: str | None
    document_ids: list[str]
    file_path: str | None
    image_path: str | None
    page_id: str | None
    page_number: int | None
    chart_id: str | None
    retrieval_mode: RetrievalMode
    top_k: int
    filters: dict[str, Any]
    metadata: dict[str, Any]


class GraphOutput(TypedDict, total=False):
    request_id: str
    task_type: TaskType
    parsed_document: ParsedDocument | None
    chart_result: ChartResult | None
    chart_answer: str | None
    chart_confidence: float | None
    graph_extraction_result: GraphExtractionResult | None
    vector_hits: list[RetrievalHit]
    graph_hits: list[RetrievalHit]
    graph_summary_hits: list[RetrievalHit]
    evidence_bundle: EvidenceBundle | None
    final_answer: QAResponse | FinalAnswer | None
    confidence: float | None
    warnings: list[str]
    reasoning_summary: dict[str, str]
    react_trace: list[dict[str, Any]]
    tool_traces: list[ToolTrace]
    metadata: dict[str, Any]
