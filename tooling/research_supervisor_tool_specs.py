from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tooling.schemas import ToolHandler, ToolSpec


SupervisorActionStatus = Literal["succeeded", "failed", "skipped"]


class SupervisorActionToolInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    invocation_id: str = Field(..., min_length=1)


class SupervisorActionToolOutput(BaseModel):
    status: SupervisorActionStatus = "succeeded"
    observation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResearchSupervisorActionDefinition:
    name: str
    description: str
    tags: tuple[str, ...]


RESEARCH_SUPERVISOR_ACTION_DEFINITIONS: dict[str, ResearchSupervisorActionDefinition] = {
    "search_literature": ResearchSupervisorActionDefinition(
        name="search_literature",
        description="Supervisor action that creates or refreshes the research task and candidate paper pool.",
        tags=("research", "supervisor_action", "discovery"),
    ),
    "write_review": ResearchSupervisorActionDefinition(
        name="write_review",
        description="Supervisor action that drafts or refreshes the current research review/report.",
        tags=("research", "supervisor_action", "writing"),
    ),
    "import_papers": ResearchSupervisorActionDefinition(
        name="import_papers",
        description="Supervisor action that imports selected papers into the local research workspace.",
        tags=("research", "supervisor_action", "import"),
    ),
    "sync_to_zotero": ResearchSupervisorActionDefinition(
        name="sync_to_zotero",
        description="Supervisor action that syncs selected candidate papers into the user's Zotero library.",
        tags=("research", "supervisor_action", "zotero"),
    ),
    "answer_question": ResearchSupervisorActionDefinition(
        name="answer_question",
        description="Supervisor action that answers a question against the current research collection.",
        tags=("research", "supervisor_action", "qa"),
    ),
    "general_answer": ResearchSupervisorActionDefinition(
        name="general_answer",
        description="Supervisor action that answers a general non-research question directly.",
        tags=("research", "supervisor_action", "general_qa"),
    ),
    "recommend_from_preferences": ResearchSupervisorActionDefinition(
        name="recommend_from_preferences",
        description="Supervisor action that recommends recent papers from the user's long-term preference memory.",
        tags=("research", "supervisor_action", "preference_recommendation"),
    ),
    "analyze_papers": ResearchSupervisorActionDefinition(
        name="analyze_papers",
        description="Supervisor action that compares or analyzes selected papers.",
        tags=("research", "supervisor_action", "analysis"),
    ),
    "compress_context": ResearchSupervisorActionDefinition(
        name="compress_context",
        description="Supervisor action that compresses selected paper context for downstream reasoning.",
        tags=("research", "supervisor_action", "context"),
    ),
    "understand_document": ResearchSupervisorActionDefinition(
        name="understand_document",
        description="Supervisor action that parses and optionally indexes an input document.",
        tags=("research", "supervisor_action", "document"),
    ),
    "supervisor_understand_chart": ResearchSupervisorActionDefinition(
        name="supervisor_understand_chart",
        description="Supervisor action that analyzes chart evidence from an input image.",
        tags=("research", "supervisor_action", "chart"),
    ),
    "analyze_paper_figures": ResearchSupervisorActionDefinition(
        name="analyze_paper_figures",
        description="Supervisor action that extracts and analyzes figures from an imported paper's PDF.",
        tags=("research", "supervisor_action", "chart", "figure"),
    ),
}


def build_research_supervisor_tool_spec(
    name: str,
    handler: ToolHandler,
    *,
    enabled: bool = True,
    max_retries: int = 0,
    strict_output_validation: bool = True,
) -> ToolSpec:
    definition = RESEARCH_SUPERVISOR_ACTION_DEFINITIONS[name]
    return ToolSpec(
        name=definition.name,
        description=definition.description,
        input_schema=SupervisorActionToolInput,
        output_schema=SupervisorActionToolOutput,
        handler=handler,
        tags=list(definition.tags),
        toolset="supervisor_action",
        category="research",
        enabled=enabled,
        max_retries=max_retries,
        strict_output_validation=strict_output_validation,
    )


class ResearchSupervisorActionRegistry:
    def __init__(self, registry) -> None:
        self.registry = registry

    def register_many(self, handlers: dict[str, ToolHandler], *, replace: bool = False) -> None:
        for name, handler in handlers.items():
            if name not in RESEARCH_SUPERVISOR_ACTION_DEFINITIONS:
                continue
            self.registry.register(
                build_research_supervisor_tool_spec(name, handler),
                replace=replace,
            )
