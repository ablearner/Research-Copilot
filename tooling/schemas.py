from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.document import ParsedDocument
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.retrieval import HybridRetrievalResult, RetrievalHit
from rag_runtime.schemas import DocumentIndexResult

ToolHandler = Callable[..., Awaitable[Any]]
ToolCallStatus = Literal[
    "succeeded",
    "failed",
    "validation_error",
    "not_found",
    "disabled",
]
ToolCategory = Literal["runtime", "research", "external_mcp"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    input_schema: type[BaseModel]
    output_schema: type[Any] | None = None
    handler: ToolHandler
    tags: list[str] = Field(default_factory=list)
    category: ToolCategory = "runtime"
    enabled: bool = True
    max_retries: int = Field(default=0, ge=0, le=3)
    timeout_seconds: float | None = Field(default=None, gt=0)
    strict_output_validation: bool = True
    audit_metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: f"call_{uuid4().hex}")
    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolAttemptTrace(BaseModel):
    attempt: int = Field(default=1, ge=1)
    status: ToolCallStatus
    error_message: str | None = None
    validation_passed: bool = True


class ToolCallTrace(BaseModel):
    call_id: str
    tool_name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: Any | None = None
    status: ToolCallStatus
    tool_category: ToolCategory = "runtime"
    attempt_count: int = Field(default=1, ge=1)
    attempts: list[ToolAttemptTrace] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0)
    error_message: str | None = None
    session_id: str | None = None
    task_id: str | None = None
    correlation_id: str | None = None
    audit_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class ToolExecutionResult(BaseModel):
    call_id: str
    tool_name: str
    status: ToolCallStatus
    output: Any | None = None
    error_message: str | None = None
    attempt_count: int = Field(default=1, ge=1)
    validation_passed: bool = True
    trace: ToolCallTrace


class ParseDocumentToolInput(BaseModel):
    file_path: str = Field(..., min_length=1)
    document_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    skill_name: str | None = None


class IndexDocumentToolInput(BaseModel):
    parsed_document: ParsedDocument
    charts: list[ChartSchema] = Field(default_factory=list)
    include_graph: bool = True
    include_embeddings: bool = True
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    skill_name: str | None = None


class UnderstandChartToolInput(BaseModel):
    image_path: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    page_id: str = Field(..., min_length=1)
    page_number: int = Field(..., ge=1)
    chart_id: str = Field(..., min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)
    skill_name: str | None = None


class HybridRetrieveToolInput(BaseModel):
    question: str = Field(..., min_length=1)
    doc_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None
    task_id: str | None = None
    memory_hints: dict[str, Any] = Field(default_factory=dict)
    skill_context: dict[str, Any] = Field(default_factory=dict)


class QueryGraphSummaryToolInput(BaseModel):
    question: str = Field(..., min_length=1)
    document_ids: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, Any] = Field(default_factory=dict)
    skill_context: dict[str, Any] = Field(default_factory=dict)


class AnswerWithEvidenceToolInput(BaseModel):
    question: str = Field(..., min_length=1)
    evidence_bundle: EvidenceBundle
    retrieval_result: HybridRetrievalResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    session_context: dict[str, Any] = Field(default_factory=dict)
    task_context: dict[str, Any] = Field(default_factory=dict)
    preference_context: dict[str, Any] = Field(default_factory=dict)
    retrieval_cache_summary: str | None = None
    memory_hints: dict[str, Any] = Field(default_factory=dict)
    skill_context: dict[str, Any] = Field(default_factory=dict)


class UnderstandChartToolOutput(BaseModel):
    chart: ChartSchema
    graph_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSummaryToolOutput(BaseModel):
    hits: list[RetrievalHit] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridRetrieveToolOutput(BaseModel):
    question: str
    document_ids: list[str] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle
    retrieval_result: HybridRetrievalResult
    metadata: dict[str, Any] = Field(default_factory=dict)


TOOL_OUTPUT_SCHEMAS: dict[str, type[Any]] = {
    "parse_document": ParsedDocument,
    "index_document": DocumentIndexResult,
    "understand_chart": UnderstandChartToolOutput,
    "hybrid_retrieve": HybridRetrieveToolOutput,
    "query_graph_summary": GraphSummaryToolOutput,
    "answer_with_evidence": QAResponse,
}
