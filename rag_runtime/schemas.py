from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.api import QAResponse
from domain.schemas.chart import ChartSchema
from domain.schemas.graph import GraphExtractionResult
from rag_runtime.services.embedding_index_service import EmbeddingIndexResult
from rag_runtime.services.graph_index_service import GraphIndexStats


class GraphTaskRequest(BaseModel):
    task_type: str
    params: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None


class GraphTaskResult(BaseModel):
    task_type: str
    status: str
    trace_id: str | None = None
    output: Any | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIndexResult(BaseModel):
    document_id: str
    graph_extraction: GraphExtractionResult | None = None
    graph_index: GraphIndexStats | None = None
    text_embedding_index: EmbeddingIndexResult | None = None
    page_embedding_index: EmbeddingIndexResult | None = None
    chart_embedding_index: EmbeddingIndexResult | None = None
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChartUnderstandingResult(BaseModel):
    chart: ChartSchema
    graph_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FusedAskResult(BaseModel):
    qa: QAResponse
    chart_answer: str | None = None
    chart_confidence: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RuntimeHealthSummary(BaseModel):
    app_name: str
    runtime_backend: str
    llm_provider: str
    embedding_provider: str
    vector_store_provider: str
    graph_store_provider: str
    graph_runtime_ready: bool
