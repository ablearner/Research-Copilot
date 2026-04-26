from typing import Any, Literal

from pydantic import BaseModel, Field


class EmbeddingItem(BaseModel):
    id: str
    document_id: str
    source_type: Literal["text_block", "page", "page_image", "chart", "image_region"]
    source_id: str
    content: str | None = None
    uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingVector(BaseModel):
    model: str
    dimensions: int = Field(..., gt=0)
    values: list[float]


class MultimodalEmbeddingRecord(BaseModel):
    id: str
    item: EmbeddingItem
    embedding: EmbeddingVector
    modality: Literal["text", "image", "chart", "page"]
    namespace: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)
