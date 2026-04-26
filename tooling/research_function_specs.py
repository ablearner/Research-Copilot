from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from domain.schemas.research_functions import RESEARCH_FUNCTION_SCHEMAS
from tooling.schemas import ToolHandler, ToolSpec


@dataclass(frozen=True, slots=True)
class ResearchFunctionDefinition:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    tags: tuple[str, ...] = ("research", "function_call")


RESEARCH_FUNCTION_DEFINITIONS: dict[str, ResearchFunctionDefinition] = {
    "search_papers": ResearchFunctionDefinition(
        name="search_papers",
        description="Search academic papers across normalized literature sources.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["search_papers"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["search_papers"][1],
        tags=("research", "search", "function_call"),
    ),
    "extract_paper_structure": ResearchFunctionDefinition(
        name="extract_paper_structure",
        description="Extract contribution, method, experiment, limitation, formulas, and figures from a paper.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["extract_paper_structure"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["extract_paper_structure"][1],
        tags=("research", "reading", "function_call"),
    ),
    "compare_papers": ResearchFunctionDefinition(
        name="compare_papers",
        description="Compare multiple papers along requested dimensions and return a structured table plus summary.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["compare_papers"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["compare_papers"][1],
        tags=("research", "comparison", "function_call"),
    ),
    "generate_review": ResearchFunctionDefinition(
        name="generate_review",
        description="Generate a structured literature review with optional citations and minimum length constraints.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["generate_review"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["generate_review"][1],
        tags=("research", "review", "function_call"),
    ),
    "ask_paper": ResearchFunctionDefinition(
        name="ask_paper",
        description="Answer a research question against specific papers and return citations and extended analysis.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["ask_paper"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["ask_paper"][1],
        tags=("research", "qa", "function_call"),
    ),
    "recommend_papers": ResearchFunctionDefinition(
        name="recommend_papers",
        description="Recommend papers based on current context and history with explicit rationales.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["recommend_papers"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["recommend_papers"][1],
        tags=("research", "recommendation", "function_call"),
    ),
    "update_research_context": ResearchFunctionDefinition(
        name="update_research_context",
        description="Update the shared ResearchContext object using new topic, goals, conclusions, and preferences.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["update_research_context"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["update_research_context"][1],
        tags=("research", "context", "function_call"),
    ),
    "decompose_task": ResearchFunctionDefinition(
        name="decompose_task",
        description="Decompose a user request into structured task steps, assign a sub-manager, and mark whether parallel execution is allowed.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["decompose_task"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["decompose_task"][1],
        tags=("research", "planning", "manager", "function_call"),
    ),
    "evaluate_result": ResearchFunctionDefinition(
        name="evaluate_result",
        description="Evaluate whether a task result satisfies the instruction and expected schema, then return issues and replanning guidance.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["evaluate_result"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["evaluate_result"][1],
        tags=("research", "evaluation", "manager", "function_call"),
    ),
    "execute_research_plan": ResearchFunctionDefinition(
        name="execute_research_plan",
        description="Execute a structured research plan and return step-level results with a summary report.",
        input_schema=RESEARCH_FUNCTION_SCHEMAS["execute_research_plan"][0],
        output_schema=RESEARCH_FUNCTION_SCHEMAS["execute_research_plan"][1],
        tags=("research", "planning", "function_call"),
    ),
}


def build_research_tool_spec(
    name: str,
    handler: ToolHandler,
    *,
    enabled: bool = True,
    max_retries: int = 3,
    tags: list[str] | None = None,
) -> ToolSpec:
    definition = RESEARCH_FUNCTION_DEFINITIONS[name]
    return ToolSpec(
        name=definition.name,
        description=definition.description,
        input_schema=definition.input_schema,
        output_schema=definition.output_schema,
        handler=handler,
        tags=tags or list(definition.tags),
        enabled=enabled,
        max_retries=max_retries,
    )


def list_research_function_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema.model_json_schema(),
            "output_schema": definition.output_schema.model_json_schema(),
            "tags": list(definition.tags),
        }
        for definition in RESEARCH_FUNCTION_DEFINITIONS.values()
    ]
