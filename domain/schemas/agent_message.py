from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from domain.schemas.research_context import ResearchContextSlice
from domain.schemas.sub_manager import TaskEvaluation


AgentTaskPriority = Literal["low", "medium", "high", "critical"]
AgentTaskStatus = Literal["planned", "running", "succeeded", "failed", "skipped"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AgentMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    agent_from: str = Field(
        validation_alias=AliasChoices("agent_from", "from"),
        serialization_alias="from",
    )
    agent_to: str = Field(
        validation_alias=AliasChoices("agent_to", "to"),
        serialization_alias="to",
    )
    task_type: str
    instruction: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    context_slice: ResearchContextSlice | dict[str, Any] = Field(default_factory=dict)
    priority: AgentTaskPriority = "medium"
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("depends_on", "dependencies"),
        serialization_alias="depends_on",
    )
    retry_count: int = Field(default=0, ge=0, le=10)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def dependencies(self) -> list[str]:
        return self.depends_on

    @dependencies.setter
    def dependencies(self, value: list[str]) -> None:
        self.depends_on = value


class AgentResultMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task_id: str
    agent_from: str = Field(
        validation_alias=AliasChoices("agent_from", "from"),
        serialization_alias="from",
    )
    agent_to: str = Field(
        validation_alias=AliasChoices("agent_to", "to"),
        serialization_alias="to",
    )
    task_type: str
    status: AgentTaskStatus
    instruction: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    context_slice: ResearchContextSlice | dict[str, Any] = Field(default_factory=dict)
    priority: AgentTaskPriority = "medium"
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("depends_on", "dependencies"),
        serialization_alias="depends_on",
    )
    retry_count: int = Field(default=0, ge=0, le=10)
    evaluation: TaskEvaluation | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def dependencies(self) -> list[str]:
        return self.depends_on

    @dependencies.setter
    def dependencies(self, value: list[str]) -> None:
        self.depends_on = value
