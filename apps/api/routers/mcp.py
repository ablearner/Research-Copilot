from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from apps.api.deps import get_graph_runtime, get_mcp_server
from apps.api.security import require_api_key
from rag_runtime.runtime import RagRuntime
from mcp.schemas import (
    MCPDiscoverySnapshot,
    MCPPromptContent,
    MCPPromptSpec,
    MCPResourceContent,
    MCPResourceSpec,
    MCPToolCallResult,
    MCPToolSpec,
)
from mcp.server.app import MCPServerApp

router = APIRouter(prefix="/mcp", tags=["mcp"])


class MCPToolCallRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None


def _get_external_tool_registry(graph_runtime: RagRuntime):
    registry = getattr(graph_runtime, "external_tool_registry", None)
    if registry is not None:
        return registry
    return getattr(graph_runtime, "mcp_client_registry", None)


@router.get("/discovery", response_model=MCPDiscoverySnapshot)
async def discovery(
    mcp_server: MCPServerApp = Depends(get_mcp_server),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
) -> MCPDiscoverySnapshot:
    local_tools = mcp_server.list_tools()
    registry = _get_external_tool_registry(graph_runtime)
    external_tools = await registry.discover_tools() if registry is not None else []
    local_tool_names = {tool.name for tool in local_tools}
    merged_tools = local_tools + [tool for tool in external_tools if tool.name not in local_tool_names]
    local_prompts = mcp_server.list_prompts()
    external_prompts = await registry.discover_prompts() if registry is not None else []
    prompt_keys = {(prompt.name, prompt.prompt_key) for prompt in local_prompts}
    merged_prompts = local_prompts + [
        prompt
        for prompt in external_prompts
        if (prompt.name, prompt.prompt_key) not in prompt_keys
    ]
    local_resources = mcp_server.list_resources()
    external_resources = await registry.discover_resources() if registry is not None else []
    resource_uris = {resource.uri for resource in local_resources}
    merged_resources = local_resources + [
        resource for resource in external_resources if resource.uri not in resource_uris
    ]
    return MCPDiscoverySnapshot(
        server=mcp_server.descriptor,
        tools=merged_tools,
        prompts=merged_prompts,
        resources=merged_resources,
    )


@router.get("/tools", response_model=list[MCPToolSpec])
async def list_tools(
    tags: list[str] | None = Query(default=None),
    names: list[str] | None = Query(default=None),
    include_disabled: bool = False,
    mcp_server: MCPServerApp = Depends(get_mcp_server),
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    _: None = Depends(require_api_key),
) -> list[MCPToolSpec]:
    tags = tags if isinstance(tags, list) else None
    names = names if isinstance(names, list) else None
    local_tools = mcp_server.list_tools(
        tags=tags,
        names=names,
        include_disabled=include_disabled,
    )
    registry = _get_external_tool_registry(graph_runtime)
    external_tools = await registry.discover_tools() if registry is not None else []
    external_tool_defs = [
        {
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
            "server_name": tool.server_name,
            "source": tool.source,
            "enabled": tool.enabled,
            "tags": tool.tags,
            "output_schema": tool.output_schema,
        }
        for tool in external_tools
        if (include_disabled or tool.enabled)
        and (not tags or set(tags) & set(tool.tags))
        and (not names or tool.name in names)
    ]
    external_specs = [
        MCPToolSpec(
            name=str(item["function"]["name"]),
            description=str(item["function"]["description"]),
            input_schema=item["function"]["parameters"],
            output_schema=item.get("output_schema"),
            tags=list(item.get("tags") or []),
            enabled=bool(item.get("enabled", True)),
            source="external",
            server_name=item.get("server_name"),
        )
        for item in external_tool_defs
    ]
    if not external_specs:
        return local_tools
    local_names = {tool.name for tool in local_tools}
    return local_tools + [tool for tool in external_specs if tool.name not in local_names]


@router.post("/tools/call", response_model=MCPToolCallResult)
async def call_tool(
    request: MCPToolCallRequest,
    graph_runtime: RagRuntime = Depends(get_graph_runtime),
    mcp_server: MCPServerApp = Depends(get_mcp_server),
    _: None = Depends(require_api_key),
) -> MCPToolCallResult:
    arguments = dict(request.arguments)
    tool_spec = graph_runtime.tool_registry.get_tool(request.tool_name, include_disabled=True)
    if tool_spec is None:
        registry = _get_external_tool_registry(graph_runtime)
        if registry is None:
            return MCPToolCallResult(
                call_id=request.call_id or f"call_missing_{request.tool_name}",
                tool_name=request.tool_name,
                status="not_found",
                output=None,
                error_message="External MCP registry is not configured.",
            )
        return await registry.call_tool(
            tool_name=request.tool_name,
            arguments=arguments,
        )
    return await mcp_server.call_tool(
        tool_name=request.tool_name,
        arguments=arguments,
        call_id=request.call_id,
    )


@router.get("/prompts", response_model=list[MCPPromptSpec])
async def list_prompts(
    mcp_server: MCPServerApp = Depends(get_mcp_server),
    _: None = Depends(require_api_key),
) -> list[MCPPromptSpec]:
    return mcp_server.list_prompts()


@router.get("/prompts/content", response_model=MCPPromptContent)
async def get_prompt(
    prompt_name: str | None = None,
    prompt_key: str | None = None,
    prompt_path: str | None = None,
    mcp_server: MCPServerApp = Depends(get_mcp_server),
    _: None = Depends(require_api_key),
) -> MCPPromptContent:
    return mcp_server.get_prompt(
        prompt_name=prompt_name,
        prompt_key=prompt_key,
        prompt_path=prompt_path,
    )


@router.get("/resources", response_model=list[MCPResourceSpec])
async def list_resources(
    mcp_server: MCPServerApp = Depends(get_mcp_server),
    _: None = Depends(require_api_key),
) -> list[MCPResourceSpec]:
    return mcp_server.list_resources()


@router.get("/resources/content", response_model=MCPResourceContent)
async def read_resource(
    uri: str,
    mcp_server: MCPServerApp = Depends(get_mcp_server),
    _: None = Depends(require_api_key),
) -> MCPResourceContent:
    return mcp_server.read_resource(uri)
