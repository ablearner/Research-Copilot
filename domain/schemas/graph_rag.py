from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.schemas.evidence import Evidence
from domain.schemas.graph import GraphEdge, GraphNode, GraphTriple


class GraphCommunity(BaseModel):
    id: str
    document_id: str
    topic: str
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    triples: list[GraphTriple] = Field(default_factory=list)
    source_references: list[Evidence] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphCommunityBuildResult(BaseModel):
    document_id: str
    communities: list[GraphCommunity] = Field(default_factory=list)
    strategy: Literal["label_topic", "source_topic", "simple"] = "label_topic"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphCommunitySummary(BaseModel):
    id: str
    community_id: str
    document_id: str
    topic: str
    summary: str
    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    source_references: list[Evidence] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphSummaryBuildResult(BaseModel):
    document_id: str
    summaries: list[GraphCommunitySummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
