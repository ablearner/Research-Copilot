from typing import Any, Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x0: float = Field(..., ge=0)
    y0: float = Field(..., ge=0)
    x1: float = Field(..., ge=0)
    y1: float = Field(..., ge=0)
    unit: Literal["pixel", "point", "relative"] = "pixel"


class TextBlock(BaseModel):
    id: str
    document_id: str
    page_id: str
    page_number: int = Field(..., ge=1)
    text: str
    bbox: BoundingBox | None = None
    block_type: Literal["paragraph", "title", "table_text", "caption", "footnote", "header", "footer"] = (
        "paragraph"
    )
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentPage(BaseModel):
    id: str
    document_id: str
    page_number: int = Field(..., ge=1)
    width: float | None = Field(default=None, gt=0)
    height: float | None = Field(default=None, gt=0)
    image_uri: str | None = None
    text_blocks: list[TextBlock] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    id: str
    filename: str
    content_type: str
    status: Literal["uploaded", "parsing", "parsed", "failed"]
    pages: list[DocumentPage] = Field(default_factory=list)
    error_message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

