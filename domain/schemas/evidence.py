from typing import Any, Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    id: str
    document_id: str | None = None
    page_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    source_type: Literal["text_block", "page_image", "chart", "graph_node", "graph_edge", "document"]
    source_id: str | None = None
    snippet: str | None = None
    score: float | None = Field(default=None, ge=0)
    graph_node_ids: list[str] = Field(default_factory=list)
    graph_edge_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    evidences: list[Evidence] = Field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

