from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from domain.schemas.research_context import QAPair, ResearchContext
from domain.schemas.sub_manager import SubManagerState, TaskStep


MemoryLayer = Literal["working", "session", "long_term", "paper_knowledge"]
MemoryStepType = Literal["plan", "retrieve", "reason", "write", "other"]
LongTermMemoryType = Literal[
    "preference",
    "annotation",
    "topic",
    "conclusion",
    "session_summary",
    "user_profile",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkingMemoryStep(BaseModel):
    step_id: str = Field(default_factory=lambda: f"step_{uuid4().hex}")
    step_type: MemoryStepType = "other"
    content: str
    tool_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class WorkingMemoryState(BaseModel):
    session_id: str
    max_turns: int = Field(default=10, ge=1, le=100)
    recent_history: list[QAPair] = Field(default_factory=list)
    selected_paper_ids: list[str] = Field(default_factory=list)
    active_paper_ids: list[str] = Field(default_factory=list)
    current_task_plan: list[TaskStep] = Field(default_factory=list)
    sub_manager_states: dict[str, SubManagerState] = Field(default_factory=dict)
    active_task_ids: list[str] = Field(default_factory=list)
    intermediate_steps: list[WorkingMemoryStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)


class ResearchSessionSummary(BaseModel):
    session_id: str
    research_topic: str = ""
    key_papers: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    summary_text: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionMemoryRecord(BaseModel):
    session_id: str
    context: ResearchContext = Field(default_factory=ResearchContext)
    read_paper_ids: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    last_task_plan: list[TaskStep] = Field(default_factory=list)
    sub_manager_states: dict[str, SubManagerState] = Field(default_factory=dict)
    summary: ResearchSessionSummary | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongTermMemoryRecord(BaseModel):
    memory_id: str = Field(default_factory=lambda: f"ltm_{uuid4().hex}")
    memory_type: LongTermMemoryType = "topic"
    topic: str = ""
    content: str = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    related_paper_ids: list[str] = Field(default_factory=list)
    source_session_id: str | None = None
    score: float | None = Field(default=None, ge=0)
    vector: list[float] = Field(default_factory=list)
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LongTermMemoryQuery(BaseModel):
    query: str = Field(min_length=1)
    topic: str | None = None
    keywords: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0)
    vector: list[float] = Field(default_factory=list)


class LongTermMemorySearchResult(BaseModel):
    query: LongTermMemoryQuery
    records: list[LongTermMemoryRecord] = Field(default_factory=list)


class InterestTopic(BaseModel):
    topic_name: str
    normalized_topic: str
    weight: float = Field(default=0.0, ge=0.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    mention_count: int = Field(default=0, ge=0)
    recent_mention_count: int = Field(default=0, ge=0)
    first_seen_at: datetime = Field(default_factory=utc_now)
    last_seen_at: datetime = Field(default_factory=utc_now)
    preferred_sources: list[str] = Field(default_factory=list)
    preferred_keywords: list[str] = Field(default_factory=list)
    preferred_recency_days: int | None = Field(default=None, ge=1, le=3650)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserResearchProfile(BaseModel):
    profile_id: str = "default"
    user_id: str = "local-user"
    display_name: str | None = None
    research_interests: list[str] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=list)
    preferred_authors: list[str] = Field(default_factory=list)
    preferred_venues: list[str] = Field(default_factory=list)
    preferred_keywords: list[str] = Field(default_factory=list)
    preferred_reasoning_style: str | None = None
    preferred_answer_language: str | None = None
    interest_topics: list[InterestTopic] = Field(default_factory=list)
    recommendation_history: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    last_active_topic: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)
