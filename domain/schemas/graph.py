from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from domain.schemas.evidence import Evidence


def _default_node_source_reference() -> Evidence:
    return Evidence(
        id=f"graph_node_src_{uuid4().hex[:12]}",
        source_type="graph_node",
        source_id=None,
        snippet=None,
    )


def _default_edge_source_reference() -> Evidence:
    return Evidence(
        id=f"graph_edge_src_{uuid4().hex[:12]}",
        source_type="graph_edge",
        source_id=None,
        snippet=None,
    )


class GraphNode(BaseModel):
    id: str
    label: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_reference: Evidence = Field(default_factory=_default_node_source_reference)


class GraphEdge(BaseModel):
    id: str
    type: str
    source_node_id: str
    target_node_id: str
    properties: dict[str, Any] = Field(default_factory=dict)
    source_reference: Evidence = Field(default_factory=_default_edge_source_reference)


class GraphTriple(BaseModel):
    subject: GraphNode
    predicate: GraphEdge
    object: GraphNode


class GraphExtractionResult(BaseModel):
    document_id: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    triples: list[GraphTriple] = Field(default_factory=list)
    status: Literal["succeeded", "partial", "failed"] = "succeeded"
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphQueryRequest(BaseModel):
    query: str
    document_ids: list[str] = Field(default_factory=list)
    node_labels: list[str] = Field(default_factory=list)
    edge_types: list[str] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=200)
    metadata_filter: dict[str, Any] = Field(default_factory=dict)


class GraphQueryResult(BaseModel):
    query: str
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    triples: list[GraphTriple] = Field(default_factory=list)
    evidences: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
