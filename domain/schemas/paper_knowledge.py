from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PaperFormulaInsight(BaseModel):
    name: str
    formula: str
    explanation: str
    purpose: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperFigureInsight(BaseModel):
    figure_id: str | None = None
    title: str = ""
    explanation: str
    purpose: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperKnowledgeCard(BaseModel):
    paper_id: str
    title: str | None = None
    contribution: str = ""
    method: str = ""
    experiment: str = ""
    limitation: str = ""
    key_formulas: list[PaperFormulaInsight] = Field(default_factory=list)
    figures: list[PaperFigureInsight] = Field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperKnowledgeRecord(BaseModel):
    paper_id: str
    document_id: str | None = None
    title: str | None = None
    core_contribution: str = ""
    user_annotations: list[str] = Field(default_factory=list)
    related_paper_ids: list[str] = Field(default_factory=list)
    citation_count: int | None = Field(default=None, ge=0)
    knowledge_card: PaperKnowledgeCard
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
