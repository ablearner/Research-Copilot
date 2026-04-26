from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.schemas.document import ParsedDocument
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.graph import GraphExtractionResult
from domain.schemas.retrieval import HybridRetrievalResult


class QARequest(BaseModel):
    question: str
    document_ids: list[str] = Field(default_factory=list)
    retrieval_mode: Literal["vector", "graph", "hybrid"] = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)


class QAResponse(BaseModel):
    answer: str
    question: str
    evidence_bundle: EvidenceBundle
    retrieval_result: HybridRetrievalResult | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UploadDocumentResponse(BaseModel):
    document_id: str
    filename: str
    status: Literal["uploaded", "failed"]
    storage_uri: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParseDocumentResponse(BaseModel):
    document_id: str
    status: Literal["parsing", "parsed", "failed"]
    parsed_document: ParsedDocument | None = None
    error_message: str | None = None


class IndexDocumentResponse(BaseModel):
    document_id: str
    status: Literal["indexing", "indexed", "failed"]
    embedding_record_count: int = Field(default=0, ge=0)
    graph_extraction: GraphExtractionResult | None = None
    error_message: str | None = None


class AskDocumentResponse(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    qa: QAResponse


class AskFusedResponse(BaseModel):
    document_ids: list[str] = Field(default_factory=list)
    qa: QAResponse
    chart_answer: str | None = None
    chart_confidence: float | None = None
