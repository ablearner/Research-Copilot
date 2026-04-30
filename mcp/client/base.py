from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mcp.schemas import MCPPromptContent, MCPPromptSpec, MCPResourceContent, MCPResourceSpec, MCPToolCallResult, MCPToolSpec

_MCP_PROTOCOL_VERSION = "2025-03-26"
_CLIENT_INFO = {"name": "research-copilot", "version": "1.0"}


def build_jsonrpc_request(
    method: str,
    params: dict[str, Any] | None = None,
    request_id: int | str | None = None,
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 request envelope."""
    msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if request_id is not None:
        msg["id"] = request_id
    return msg


class BaseMCPClient(ABC):
    _initialized: bool = False
    _server_info: dict[str, Any] | None = None
    _server_capabilities: dict[str, Any] | None = None

    @property
    @abstractmethod
    def server_name(self) -> str:
        raise NotImplementedError

    async def connect(self) -> None:
        """Establish transport connection (override in subclasses)."""

    async def disconnect(self) -> None:
        """Tear down transport connection (override in subclasses)."""

    async def initialize(self) -> dict[str, Any]:
        """Perform MCP initialize handshake.

        Returns the server's initialize result dict, or empty dict on failure.
        """
        return {}

    @abstractmethod
    async def list_tools(self) -> list[MCPToolSpec]:
        raise NotImplementedError

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> MCPToolCallResult:
        raise NotImplementedError

    @abstractmethod
    async def list_prompts(self) -> list[MCPPromptSpec]:
        raise NotImplementedError

    @abstractmethod
    async def get_prompt(
        self,
        *,
        prompt_name: str | None = None,
        prompt_key: str | None = None,
        skill_name: str | None = None,
    ) -> MCPPromptContent:
        raise NotImplementedError

    @abstractmethod
    async def list_resources(self) -> list[MCPResourceSpec]:
        raise NotImplementedError

    @abstractmethod
    async def read_resource(self, uri: str) -> MCPResourceContent:
        raise NotImplementedError


class InProcessMCPClient(BaseMCPClient):
    def __init__(self, app) -> None:
        self._app = app
        self._initialized = True

    @property
    def server_name(self) -> str:
        return self._app.descriptor.name

    async def list_tools(self) -> list[MCPToolSpec]:
        return self._app.list_tools()

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> MCPToolCallResult:
        return await self._app.call_tool(tool_name=tool_name, arguments=arguments, call_id=call_id)

    async def list_prompts(self) -> list[MCPPromptSpec]:
        return self._app.list_prompts()

    async def get_prompt(
        self,
        *,
        prompt_name: str | None = None,
        prompt_key: str | None = None,
        skill_name: str | None = None,
    ) -> MCPPromptContent:
        return self._app.get_prompt(prompt_name=prompt_name, prompt_key=prompt_key, skill_name=skill_name)

    async def list_resources(self) -> list[MCPResourceSpec]:
        return self._app.list_resources()

    async def read_resource(self, uri: str) -> MCPResourceContent:
        return self._app.read_resource(uri)
