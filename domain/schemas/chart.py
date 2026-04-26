from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from domain.schemas.document import BoundingBox


class AxisSchema(BaseModel):
    name: str | None = None
    label: str | None = None
    unit: str | None = None
    scale: Literal["linear", "log", "time", "categorical", "unknown"] = "unknown"
    min_value: float | str | None = None
    max_value: float | str | None = None
    categories: list[str] = Field(default_factory=list)

    @field_validator("categories", mode="before")
    @classmethod
    def _normalize_categories(cls, value: object) -> object:
        # Some vision providers return null for optional arrays.
        return [] if value is None else value


class SeriesPoint(BaseModel):
    x: float | str | None = None
    y: float | str | None = None
    value: float | str | None = None
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SeriesSchema(BaseModel):
    name: str
    chart_role: Literal["bar", "line", "scatter", "area", "pie_slice", "table_cell", "unknown"] = "unknown"
    points: list[SeriesPoint] = Field(default_factory=list)
    unit: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChartSchema(BaseModel):
    id: str
    document_id: str
    page_id: str
    page_number: int = Field(..., ge=1)
    chart_type: Literal["bar", "line", "scatter", "pie", "table", "mixed", "unknown"] = "unknown"
    title: str | None = None
    caption: str | None = None
    bbox: BoundingBox | None = None
    x_axis: AxisSchema | None = None
    y_axis: AxisSchema | None = None
    series: list[SeriesSchema] = Field(default_factory=list)
    summary: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

