from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: int
    action: str
    instruction: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"
    task_id: str | None = None
    attempts: int = 0
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    allow_skip: bool = False


class ResearchPlan(BaseModel):
    plan_id: str
    revision: int = 1
    replan_attempts: int = 0
    steps: list[PlanStep]
    status: Literal["active", "completed", "blocked", "needs_replan", "failed"] = "active"
    reason: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)
