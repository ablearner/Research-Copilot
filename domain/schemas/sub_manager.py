from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


TaskPriority = Literal["low", "normal", "high"]
TaskExecutionMode = Literal["serial", "parallel"]
TaskStepStatus = Literal["planned", "queued", "running", "succeeded", "failed", "skipped"]
SubManagerStatus = Literal["idle", "running", "blocked", "completed", "failed"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStep(BaseModel):
    task_id: str
    assigned_to: str
    instruction: str = ""
    task_type: str = "research"
    depends_on: list[str] = Field(default_factory=list)
    context_slice: dict[str, Any] = Field(default_factory=dict)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    execution_mode: TaskExecutionMode = "serial"
    priority: TaskPriority = "normal"
    retry_count: int = Field(default=0, ge=0, le=10)
    status: TaskStepStatus = "planned"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskEvaluation(BaseModel):
    passed: bool = False
    score: float = Field(default=0.0, ge=0.0, le=10.0)
    issues: list[str] = Field(default_factory=list)
    replan_suggestion: str | None = None
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SubManagerState(BaseModel):
    name: str
    status: SubManagerStatus = "idle"
    active_task_ids: list[str] = Field(default_factory=list)
    completed_task_ids: list[str] = Field(default_factory=list)
    last_task_plan_id: str | None = None
    last_evaluation: TaskEvaluation | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utc_now)
