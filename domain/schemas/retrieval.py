from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.schemas.evidence import EvidenceBundle
from domain.schemas.graph import GraphEdge, GraphNode, GraphTriple


class RetrievalQuery(BaseModel):
    query: str
    document_ids: list[str] = Field(default_factory=list)
    mode: Literal["vector", "graph", "sparse", "hybrid"] = "hybrid"
    modalities: list[Literal["text", "image", "chart", "page"]] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] = Field(default_factory=dict)
    graph_query_mode: Literal["entity", "subgraph", "summary", "auto"] = "auto"


class RetrievalHit(BaseModel):
    id: str
    source_type: Literal[
        "text_block",
        "page",
        "page_image",
        "chart",
        "image_region",
        "graph_node",
        "graph_edge",
        "graph_triple",
        "graph_subgraph",
        "graph_summary",
    ]
    source_id: str
    document_id: str | None = None
    content: str | None = None
    sparse_score: float | None = Field(default=None, ge=0)
    vector_score: float | None = Field(default=None, ge=0)
    graph_score: float | None = Field(default=None, ge=0)
    # Cross-encoder rerank scores are ordering scores, not probabilities, and may be negative.
    merged_score: float | None = None
    graph_nodes: list[GraphNode] = Field(default_factory=list)
    graph_edges: list[GraphEdge] = Field(default_factory=list)
    graph_triples: list[GraphTriple] = Field(default_factory=list)
    evidence: EvidenceBundle | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridRetrievalResult(BaseModel):
    query: RetrievalQuery
    hits: list[RetrievalHit] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    metadata: dict[str, Any] = Field(default_factory=dict)
