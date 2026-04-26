from __future__ import annotations

from typing import Any

from mcp.schemas import MCPPromptSpec, MCPResourceSpec, MCPToolSpec
from tooling.schemas import ToolSpec


def map_tool_spec_to_mcp_tool(tool_spec: ToolSpec, source: str = "local", server_name: str | None = None) -> MCPToolSpec:
    output_schema = None
    if tool_spec.output_schema and hasattr(tool_spec.output_schema, "model_json_schema"):
        output_schema = tool_spec.output_schema.model_json_schema()
    return MCPToolSpec(
        name=tool_spec.name,
        description=tool_spec.description,
        input_schema=tool_spec.input_schema.model_json_schema(),
        output_schema=output_schema,
        tags=tool_spec.tags,
        enabled=tool_spec.enabled,
        source=source if source in {"local", "external"} else "local",
        server_name=server_name,
    )


def map_tool_specs_to_mcp_tools(
    tool_specs: list[ToolSpec],
    source: str = "local",
    server_name: str | None = None,
) -> list[MCPToolSpec]:
    return [map_tool_spec_to_mcp_tool(tool_spec, source=source, server_name=server_name) for tool_spec in tool_specs]


def map_prompt_to_mcp_prompt(
    *,
    name: str,
    prompt_key: str,
    path: str,
    skill_name: str | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MCPPromptSpec:
    return MCPPromptSpec(
        name=name,
        description=description or f"Prompt template for {prompt_key}",
        prompt_key=prompt_key,
        path=path,
        skill_name=skill_name,
        metadata=metadata or {},
    )


def make_resource_uri(resource_type: str, resource_id: str) -> str:
    return f"mcp://resource/{resource_type}/{resource_id}"


def map_resource_to_mcp_resource(
    *,
    resource_type: str,
    resource_id: str,
    name: str,
    description: str,
    mime_type: str = "application/json",
    metadata: dict[str, Any] | None = None,
) -> MCPResourceSpec:
    return MCPResourceSpec(
        uri=make_resource_uri(resource_type, resource_id),
        name=name,
        description=description,
        resource_type=(
            resource_type
            if resource_type in {"document_summary", "chart_summary", "graph_community_summary", "schema", "config", "custom"}
            else "custom"
        ),
        mime_type=mime_type,
        read_only=True,
        metadata=metadata or {},
    )
