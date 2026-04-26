from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from mcp.client.base import BaseMCPClient
from mcp.schemas import MCPPromptSpec, MCPResourceSpec, MCPToolCallResult, MCPToolSpec

logger = logging.getLogger(__name__)


class MCPClientRegistryError(RuntimeError):
    """Raised when MCP client registry operations fail."""


class MCPClientRegistry:
    def __init__(self) -> None:
        self._clients: dict[str, BaseMCPClient] = {}

    def register_server(self, name: str, client: BaseMCPClient, replace: bool = False) -> None:
        if name in self._clients and not replace:
            raise MCPClientRegistryError(f"MCP server already registered: {name}")
        self._clients[name] = client
        logger.info("MCP server registered", extra={"server_name": name})

    def unregister_server(self, name: str) -> None:
        self._clients.pop(name, None)

    def list_servers(self) -> list[str]:
        return sorted(self._clients.keys())

    def get_server(self, name: str) -> BaseMCPClient | None:
        return self._clients.get(name)

    async def discover_tools(self, server_name: str | None = None) -> list[MCPToolSpec]:
        clients = self._target_clients(server_name)
        tools: list[MCPToolSpec] = []
        for name, client in clients:
            discovered = await client.list_tools()
            for tool in discovered:
                tools.append(tool.model_copy(update={"server_name": name, "source": "external"}))
        return tools

    async def discover_prompts(self, server_name: str | None = None) -> list[MCPPromptSpec]:
        clients = self._target_clients(server_name)
        prompts: list[MCPPromptSpec] = []
        for _name, client in clients:
            prompts.extend(await client.list_prompts())
        return prompts

    async def discover_resources(self, server_name: str | None = None) -> list[MCPResourceSpec]:
        clients = self._target_clients(server_name)
        resources: list[MCPResourceSpec] = []
        for _name, client in clients:
            resources.extend(await client.list_resources())
        return resources

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        server_name: str | None = None,
        call_id: str | None = None,
    ) -> MCPToolCallResult:
        client = await self._resolve_client_for_tool(tool_name=tool_name, server_name=server_name)
        if client is None:
            return MCPToolCallResult(
                call_id=call_id or f"call_{uuid4().hex}",
                tool_name=tool_name,
                status="not_found",
                output=None,
                error_message="External MCP tool not found",
                server_name=server_name,
            )
        return await client.call_tool(tool_name=tool_name, arguments=arguments, call_id=call_id)

    async def _resolve_client_for_tool(
        self,
        tool_name: str,
        server_name: str | None,
    ) -> BaseMCPClient | None:
        if server_name:
            return self._clients.get(server_name)

        for _name, client in self._clients.items():
            tools = await client.list_tools()
            if any(tool.name == tool_name and tool.enabled for tool in tools):
                return client
        return None

    def _target_clients(self, server_name: str | None) -> list[tuple[str, BaseMCPClient]]:
        if server_name:
            client = self._clients.get(server_name)
            if client is None:
                return []
            return [(server_name, client)]
        return list(self._clients.items())
