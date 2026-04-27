"""MCP client that connects to external MCP servers via HTTP/SSE transport."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.client.base import BaseMCPClient
from mcp.schemas import (
    MCPPromptContent,
    MCPPromptSpec,
    MCPResourceContent,
    MCPResourceSpec,
    MCPToolCallResult,
    MCPToolSpec,
)
from mcp.security import sanitize_error

logger = logging.getLogger(__name__)


class HttpMCPClient(BaseMCPClient):
    """Connect to an external MCP server via HTTP (Streamable HTTP / SSE).

    Features:
    - Auto-retry with exponential backoff
    - Timeout control per tool call
    - Error sanitization
    """

    def __init__(
        self,
        server_name: str,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        tool_timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self._server_name = server_name
        self._base_url = base_url.rstrip("/")
        self._headers = headers or {}
        self._tool_timeout = tool_timeout
        self._max_retries = max_retries
        self._tools_cache: list[MCPToolSpec] | None = None

    @property
    def server_name(self) -> str:
        return self._server_name

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC style POST to the MCP HTTP server."""
        import httpx

        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._tool_timeout, headers=self._headers) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def list_tools(self) -> list[MCPToolSpec]:
        if self._tools_cache is not None:
            return self._tools_cache

        resp = await self._post("/mcp", {"method": "tools/list", "params": {}})
        tools = [
            MCPToolSpec(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
                source="external",
                server_name=self._server_name,
            )
            for t in resp.get("result", {}).get("tools", [])
        ]
        self._tools_cache = tools
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> MCPToolCallResult:
        from uuid import uuid4

        cid = call_id or f"call_{uuid4().hex}"

        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._post("/mcp", {
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments or {}},
                })
                result = resp.get("result", {})
                is_error = result.get("isError", False)
                content_parts = result.get("content", [])
                text_output = "\n".join(
                    p.get("text", "") for p in content_parts if p.get("type") == "text"
                )
                return MCPToolCallResult(
                    call_id=cid,
                    tool_name=tool_name,
                    status="failed" if is_error else "succeeded",
                    output=text_output or None,
                    error_message=sanitize_error(text_output) if is_error else None,
                    server_name=self._server_name,
                )
            except Exception as exc:
                if attempt < self._max_retries:
                    delay = min(2 ** attempt, 15)
                    logger.warning(
                        "MCP HTTP call failed (attempt %d/%d): %s",
                        attempt + 1, self._max_retries + 1, sanitize_error(str(exc)),
                    )
                    await asyncio.sleep(delay)
                else:
                    return MCPToolCallResult(
                        call_id=cid,
                        tool_name=tool_name,
                        status="failed",
                        error_message=sanitize_error(str(exc)),
                        server_name=self._server_name,
                    )

        return MCPToolCallResult(
            call_id=cid, tool_name=tool_name, status="failed",
            error_message="Exhausted retries", server_name=self._server_name,
        )

    async def list_prompts(self) -> list[MCPPromptSpec]:
        return []

    async def get_prompt(self, *, prompt_name=None, prompt_key=None, skill_name=None) -> MCPPromptContent:
        raise NotImplementedError("Prompts not supported over HTTP transport")

    async def list_resources(self) -> list[MCPResourceSpec]:
        return []

    async def read_resource(self, uri: str) -> MCPResourceContent:
        raise NotImplementedError("Resources not supported over HTTP transport")
