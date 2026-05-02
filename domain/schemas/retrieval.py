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


def retrieval_hit_score(hit: RetrievalHit) -> float:
    return float(hit.merged_score or hit.vector_score or hit.graph_score or 0.0)


def merge_retrieval_hits(*groups: list[RetrievalHit]) -> list[RetrievalHit]:
    merged: dict[tuple[str, str, str | None, str | None], RetrievalHit] = {}
    for group in groups:
        for hit in group:
            key = (
                hit.id,
                hit.source_id,
                hit.document_id,
                (hit.content or "")[:240] or None,
            )
            existing = merged.get(key)
            if existing is None or retrieval_hit_score(hit) >= retrieval_hit_score(existing):
                merged[key] = hit
    return sorted(merged.values(), key=retrieval_hit_score, reverse=True)


class HybridRetrievalResult(BaseModel):
    query: RetrievalQuery
    hits: list[RetrievalHit] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    metadata: dict[str, Any] = Field(default_factory=dict)
