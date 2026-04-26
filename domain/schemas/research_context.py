from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from domain.schemas.research import PaperSource
from domain.schemas.sub_manager import SubManagerState, TaskStep


ResearchReviewStyle = Literal["academic", "concise", "beginner"]
PaperSummaryLevel = Literal["paragraph", "section", "document"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ResearchUserPreferences(BaseModel):
    review_style: ResearchReviewStyle = "academic"
    preferred_sources: list[PaperSource | str] = Field(default_factory=list)
    answer_language: str = "zh-CN"
    citation_style: str = "inline_brackets"
    max_selected_papers: int = Field(default=8, ge=1, le=50)
    max_history_turns: int = Field(default=10, ge=1, le=50)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchContextPaperMeta(BaseModel):
    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    source: PaperSource | str | None = None
    document_id: str | None = None
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAPair(BaseModel):
    question: str
    answer: str
    citations: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompressedPaperSummary(BaseModel):
    paper_id: str
    level: PaperSummaryLevel = "document"
    summary: str
    source_section_ids: list[str] = Field(default_factory=list)
    relevance_score: float | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


def default_sub_manager_states() -> dict[str, SubManagerState]:
    return {
        "writing": SubManagerState(name="writing"),
        "research": SubManagerState(name="research"),
    }


class ResearchContext(BaseModel):
    research_topic: str = ""
    research_goals: list[str] = Field(default_factory=list)
    selected_papers: list[str] = Field(default_factory=list)
    active_papers: list[str] = Field(default_factory=list)
    imported_papers: list[ResearchContextPaperMeta] = Field(default_factory=list)
    known_conclusions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    session_history: list[QAPair] = Field(default_factory=list)
    user_preferences: ResearchUserPreferences = Field(default_factory=ResearchUserPreferences)
    paper_summaries: list[CompressedPaperSummary] = Field(default_factory=list)
    current_task_plan: list[TaskStep] = Field(default_factory=list)
    sub_manager_states: dict[str, SubManagerState] = Field(default_factory=default_sub_manager_states)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class ResearchContextSlice(BaseModel):
    research_topic: str = ""
    research_goals: list[str] = Field(default_factory=list)
    selected_papers: list[str] = Field(default_factory=list)
    imported_papers: list[ResearchContextPaperMeta] = Field(default_factory=list)
    known_conclusions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    session_history: list[QAPair] = Field(default_factory=list)
    relevant_summaries: list[CompressedPaperSummary] = Field(default_factory=list)
    current_task_plan: list[TaskStep] = Field(default_factory=list)
    sub_manager_state: SubManagerState | None = None
    context_scope: Literal["manager", "sub_manager", "worker"] = "worker"
    summary_level: PaperSummaryLevel = "document"
    user_preferences: ResearchUserPreferences | None = None
    memory_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)
