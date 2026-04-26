from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.research import PaperCandidate
from mcp.server.app import MCPServerApp
from mcp.server.prompt_adapter import MCPPromptAdapter
from mcp.server.resource_adapter import MCPResourceAdapter
from mcp.server.tool_adapter import MCPToolAdapter
from core.prompt_resolver import PromptResolver
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from tooling.schemas import ToolSpec
from tools.research import (
    ArxivSearchTool,
    IEEEMetadataSearchTool,
    OpenAlexSearchTool,
    SemanticScholarSearchTool,
)


class AcademicSearchToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    max_results: int = Field(default=10, ge=1, le=50)
    days_back: int = Field(default=90, ge=1, le=3650)


class AcademicSearchToolOutput(BaseModel):
    papers: list[PaperCandidate] = Field(default_factory=list)


@dataclass(slots=True)
class AcademicSearchMCPDependencies:
    arxiv_tool: ArxivSearchTool
    openalex_tool: OpenAlexSearchTool
    semantic_scholar_tool: SemanticScholarSearchTool | None = None
    ieee_tool: IEEEMetadataSearchTool | None = None


def build_academic_search_mcp_app(
    dependencies: AcademicSearchMCPDependencies,
    *,
    server_name: str = "academic-search",
) -> MCPServerApp:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)

    async def arxiv_search(**kwargs: Any) -> AcademicSearchToolOutput:
        papers = await dependencies.arxiv_tool.search(**kwargs)
        return AcademicSearchToolOutput(papers=papers)

    async def openalex_search(**kwargs: Any) -> AcademicSearchToolOutput:
        papers = await dependencies.openalex_tool.search(**kwargs)
        return AcademicSearchToolOutput(papers=papers)

    async def semantic_scholar_search(**kwargs: Any) -> AcademicSearchToolOutput:
        if dependencies.semantic_scholar_tool is None:
            raise RuntimeError("Semantic Scholar MCP tool is not configured")
        papers = await dependencies.semantic_scholar_tool.search(**kwargs)
        return AcademicSearchToolOutput(papers=papers)

    async def ieee_search(**kwargs: Any) -> AcademicSearchToolOutput:
        if dependencies.ieee_tool is None:
            raise RuntimeError("IEEE MCP tool is not configured")
        papers = await dependencies.ieee_tool.search(**kwargs)
        return AcademicSearchToolOutput(papers=papers)

    registry.register_many(
        [
            ToolSpec(
                name="academic_search_arxiv",
                description="Search arXiv and return normalized paper candidates.",
                input_schema=AcademicSearchToolInput,
                output_schema=AcademicSearchToolOutput,
                handler=arxiv_search,
                tags=["academic", "mcp", "search", "arxiv"],
            ),
            ToolSpec(
                name="academic_search_openalex",
                description="Search OpenAlex and return normalized paper candidates.",
                input_schema=AcademicSearchToolInput,
                output_schema=AcademicSearchToolOutput,
                handler=openalex_search,
                tags=["academic", "mcp", "search", "openalex"],
            ),
            ToolSpec(
                name="academic_search_semantic_scholar",
                description="Search Semantic Scholar and return normalized paper candidates.",
                input_schema=AcademicSearchToolInput,
                output_schema=AcademicSearchToolOutput,
                handler=semantic_scholar_search,
                tags=["academic", "mcp", "search", "semantic_scholar"],
                enabled=dependencies.semantic_scholar_tool is not None,
            ),
            ToolSpec(
                name="academic_search_ieee",
                description="Search IEEE Xplore metadata and return normalized paper candidates.",
                input_schema=AcademicSearchToolInput,
                output_schema=AcademicSearchToolOutput,
                handler=ieee_search,
                tags=["academic", "mcp", "search", "ieee"],
                enabled=dependencies.ieee_tool is not None,
            ),
        ],
        replace=True,
    )
    resource_adapter = MCPResourceAdapter()
    resource_adapter.set_config_info(
        "academic_search",
        {
            "server_name": server_name,
            "tools": [
                "academic_search_arxiv",
                "academic_search_openalex",
                "academic_search_semantic_scholar",
                "academic_search_ieee",
            ],
        },
    )
    return MCPServerApp(
        server_name=server_name,
        description="Academic search MCP server",
        tool_adapter=MCPToolAdapter(registry=registry, executor=executor, server_name=server_name),
        prompt_adapter=MCPPromptAdapter(prompt_resolver=PromptResolver()),
        resource_adapter=resource_adapter,
    )


def build_academic_search_mcp_client(dependencies: AcademicSearchMCPDependencies):
    from mcp.client.base import InProcessMCPClient

    return InProcessMCPClient(build_academic_search_mcp_app(dependencies))
