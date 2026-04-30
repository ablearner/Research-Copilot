"""MCP client that connects to external MCP servers via HTTP/SSE transport."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.client.base import BaseMCPClient, build_jsonrpc_request, _MCP_PROTOCOL_VERSION, _CLIENT_INFO
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
        self._initialized = False
        self._request_id = 0

    @property
    def server_name(self) -> str:
        return self._server_name

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 POST to the MCP HTTP server."""
        import httpx

        url = f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=self._tool_timeout, headers=self._headers) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                err = data["error"]
                raise RuntimeError(f"JSON-RPC error {err.get('code', -1)}: {err.get('message', 'unknown')}")
            return data

    async def initialize(self) -> dict[str, Any]:
        """Perform MCP initialize handshake over HTTP."""
        if self._initialized:
            return {"serverInfo": self._server_info or {}, "capabilities": self._server_capabilities or {}}
        try:
            req = build_jsonrpc_request(
                "initialize",
                params={
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "clientInfo": _CLIENT_INFO,
                },
                request_id=self._next_id(),
            )
            resp = await self._post("/mcp", req)
            result = resp.get("result", {})
            self._server_info = result.get("serverInfo", {})
            self._server_capabilities = result.get("capabilities", {})
            self._initialized = True
            logger.info(
                "MCP HTTP initialized: %s (server=%s)",
                self._server_name,
                self._server_info.get("name", "unknown"),
            )
            return result
        except Exception as exc:
            logger.warning(
                "MCP initialize handshake failed for %s, continuing without: %s",
                self._server_name, sanitize_error(str(exc)),
            )
            self._initialized = True
            return {}

    async def list_tools(self) -> list[MCPToolSpec]:
        if self._tools_cache is not None:
            return self._tools_cache

        if not self._initialized:
            await self.initialize()

        req = build_jsonrpc_request("tools/list", params={}, request_id=self._next_id())
        resp = await self._post("/mcp", req)
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
                req = build_jsonrpc_request(
                    "tools/call",
                    params={"name": tool_name, "arguments": arguments or {}},
                    request_id=self._next_id(),
                )
                resp = await self._post("/mcp", req)
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
