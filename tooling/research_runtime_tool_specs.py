from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from tooling.research_runtime_schemas import (
    AcademicSearchToolInput,
    CodeExecutionToolInput,
    CodeExecutionToolOutput,
    LibrarySyncToolInput,
    LibrarySyncToolOutput,
    LocalFileToolInput,
    LocalFileToolOutput,
    NotificationToolInput,
    NotificationToolOutput,
    SearchOrImportPaperToolInput,
    SearchOrImportPaperToolOutput,
    WebSearchToolInput,
    WebSearchToolOutput,
)
from domain.schemas.research_functions import SearchPapersFunctionOutput
from tooling.schemas import ToolHandler, ToolSpec


@dataclass(frozen=True, slots=True)
class ResearchRuntimeToolDefinition:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[Any]
    tags: tuple[str, ...]


RESEARCH_RUNTIME_TOOL_DEFINITIONS: dict[str, ResearchRuntimeToolDefinition] = {
    "academic_search": ResearchRuntimeToolDefinition(
        name="academic_search",
        description="Search academic papers via the normalized research search pipeline.",
        input_schema=AcademicSearchToolInput,
        output_schema=SearchPapersFunctionOutput,
        tags=("research", "local_runtime", "search"),
    ),
    "local_file": ResearchRuntimeToolDefinition(
        name="local_file",
        description="Read, write, append, delete, or list research-related local files.",
        input_schema=LocalFileToolInput,
        output_schema=LocalFileToolOutput,
        tags=("research", "local_runtime", "filesystem"),
    ),
    "code_execution": ResearchRuntimeToolDefinition(
        name="code_execution",
        description="Execute Python code in a bounded subprocess and return stdout/stderr.",
        input_schema=CodeExecutionToolInput,
        output_schema=CodeExecutionToolOutput,
        tags=("research", "local_runtime", "execution"),
    ),
    "web_search": ResearchRuntimeToolDefinition(
        name="web_search",
        description="Search technical web sources via Tavily or Brave when configured.",
        input_schema=WebSearchToolInput,
        output_schema=WebSearchToolOutput,
        tags=("research", "local_runtime", "web"),
    ),
    "notification": ResearchRuntimeToolDefinition(
        name="notification",
        description="Enqueue, list, or dismiss research notifications.",
        input_schema=NotificationToolInput,
        output_schema=NotificationToolOutput,
        tags=("research", "local_runtime", "notification"),
    ),
    "library_sync": ResearchRuntimeToolDefinition(
        name="library_sync",
        description="Export or sync paper metadata to filesystem, Zotero, or Notion targets.",
        input_schema=LibrarySyncToolInput,
        output_schema=LibrarySyncToolOutput,
        tags=("research", "local_runtime", "sync"),
    ),
    "search_or_import_paper": ResearchRuntimeToolDefinition(
        name="search_or_import_paper",
        description="Search papers online, deduplicate against local Zotero, and import only when missing.",
        input_schema=SearchOrImportPaperToolInput,
        output_schema=SearchOrImportPaperToolOutput,
        tags=("research", "local_runtime", "workflow", "zotero"),
    ),
}


def build_research_runtime_tool_spec(
    name: str,
    handler: ToolHandler,
    *,
    enabled: bool = True,
    max_retries: int = 2,
    strict_output_validation: bool = True,
) -> ToolSpec:
    definition = RESEARCH_RUNTIME_TOOL_DEFINITIONS[name]
    return ToolSpec(
        name=definition.name,
        description=definition.description,
        input_schema=definition.input_schema,
        output_schema=definition.output_schema,
        handler=handler,
        tags=list(definition.tags),
        toolset="runtime_ops",
        enabled=enabled,
        max_retries=max_retries,
        strict_output_validation=strict_output_validation,
    )
