"""Serializable prompt data for one specialist invocation."""
from pydantic import BaseModel, Field, JsonValue

from domain.schemas.research_context import ResearchContextSlice


class WorkerArtifactReference(BaseModel):
    key: str
    task_id: str
    kind: str = "result"
    summary: str | None = None


class WorkerTaskInput(BaseModel):
    task_id: str | None = None
    action_name: str
    instruction: str
    context_slice: ResearchContextSlice = Field(default_factory=ResearchContextSlice)
    dependency_summaries: list[dict[str, JsonValue]] = Field(default_factory=list)
    artifact_refs: list[WorkerArtifactReference] = Field(default_factory=list)
