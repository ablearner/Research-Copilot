from __future__ import annotations

from typing import Any

from mcp.schemas import (
    MCPPromptContent,
    MCPPromptSpec,
    MCPResourceContent,
    MCPResourceSpec,
    MCPServerDescriptor,
    MCPToolCallResult,
    MCPToolSpec,
)
from mcp.server.prompt_adapter import MCPPromptAdapter
from mcp.server.resource_adapter import MCPResourceAdapter
from mcp.server.tool_adapter import MCPToolAdapter


class MCPServerApp:
    def __init__(
        self,
        server_name: str,
        tool_adapter: MCPToolAdapter,
        prompt_adapter: MCPPromptAdapter,
        resource_adapter: MCPResourceAdapter,
        description: str | None = None,
    ) -> None:
        self.descriptor = MCPServerDescriptor(
            name=server_name,
            description=description or "Research-Copilot MCP Server",
            enabled=True,
        )
        self.tool_adapter = tool_adapter
        self.prompt_adapter = prompt_adapter
        self.resource_adapter = resource_adapter

    @classmethod
    def from_graph_runtime(
        cls,
        graph_runtime,
        server_name: str = "research-copilot-local",
        description: str | None = None,
    ):
        tool_adapter = MCPToolAdapter(
            registry=graph_runtime.tool_registry,
            executor=graph_runtime.tool_executor,
            server_name=server_name,
        )
        prompt_adapter = MCPPromptAdapter(prompt_resolver=graph_runtime.prompt_resolver)
        resource_adapter = MCPResourceAdapter()
        cls._bootstrap_system_resources(graph_runtime, resource_adapter)
        return cls(
            server_name=server_name,
            tool_adapter=tool_adapter,
            prompt_adapter=prompt_adapter,
            resource_adapter=resource_adapter,
            description=description,
        )

    @staticmethod
    def _bootstrap_system_resources(graph_runtime, resource_adapter: MCPResourceAdapter) -> None:
        tool_schemas: dict[str, Any] = {}
        for tool_spec in graph_runtime.tool_registry.list_tools(include_disabled=True):
            output_schema = None
            if tool_spec.output_schema and hasattr(tool_spec.output_schema, "model_json_schema"):
                output_schema = tool_spec.output_schema.model_json_schema()
            tool_schemas[tool_spec.name] = {
                "description": tool_spec.description,
                "enabled": tool_spec.enabled,
                "tags": tool_spec.tags,
                "input_schema": tool_spec.input_schema.model_json_schema(),
                "output_schema": output_schema,
            }
        resource_adapter.set_schema_info("tool_schemas", {"tools": tool_schemas})

        prompt_mapping = graph_runtime.prompt_resolver.load_mapping()
        resource_adapter.set_config_info("prompt_mapping", prompt_mapping)

    def list_tools(
        self,
        tags: list[str] | None = None,
        names: list[str] | None = None,
        include_disabled: bool = False,
    ) -> list[MCPToolSpec]:
        return self.tool_adapter.list_tools(
            tags=tags,
            names=names,
            include_disabled=include_disabled,
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> MCPToolCallResult:
        return await self.tool_adapter.call_tool(tool_name=tool_name, arguments=arguments, call_id=call_id)

    def list_prompts(self) -> list[MCPPromptSpec]:
        return self.prompt_adapter.list_prompts()

    def get_prompt(
        self,
        *,
        prompt_name: str | None = None,
        prompt_key: str | None = None,
        prompt_path: str | None = None,
    ) -> MCPPromptContent:
        return self.prompt_adapter.get_prompt(
            prompt_name=prompt_name,
            prompt_key=prompt_key,
            prompt_path=prompt_path,
        )

    def list_resources(self) -> list[MCPResourceSpec]:
        return self.resource_adapter.list_resources()

    def read_resource(self, uri: str) -> MCPResourceContent:
        return self.resource_adapter.read_resource(uri)
