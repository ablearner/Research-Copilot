"""MCP client that connects to external MCP servers via stdio transport."""

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
from mcp.security import build_safe_env, sanitize_error

logger = logging.getLogger(__name__)


class StdioMCPClient(BaseMCPClient):
    """Connect to an external MCP server via stdio subprocess transport.

    Features:
    - Safe env filtering (only whitelisted keys passed to subprocess)
    - Auto-reconnect with exponential backoff
    - Timeout control per tool call
    - Error sanitization (credentials stripped from error messages)
    """

    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        tool_timeout: int = 120,
        connect_timeout: int = 60,
        max_retries: int = 5,
    ) -> None:
        self._server_name = server_name
        self._command = command
        self._args = args or []
        self._user_env = env
        self._tool_timeout = tool_timeout
        self._connect_timeout = connect_timeout
        self._max_retries = max_retries
        self._process: asyncio.subprocess.Process | None = None
        self._connected = False
        self._initialized = False
        self._tools_cache: list[MCPToolSpec] | None = None
        self._request_id = 0

    @property
    def server_name(self) -> str:
        return self._server_name

    async def connect(self) -> None:
        """Start subprocess and establish MCP session."""
        safe_env = build_safe_env(self._user_env)
        try:
            self._process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    self._command,
                    *self._args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=safe_env,
                ),
                timeout=self._connect_timeout,
            )
            self._connected = True
            logger.info("MCP stdio server connected: %s (pid=%s)", self._server_name, self._process.pid)
        except asyncio.TimeoutError:
            raise ConnectionError(f"MCP stdio connect timeout ({self._connect_timeout}s): {self._server_name}")
        except FileNotFoundError:
            raise ConnectionError(f"MCP stdio command not found: {self._command}")

    async def disconnect(self) -> None:
        """Terminate the subprocess."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
        self._connected = False
        self._tools_cache = None

    async def _reconnect(self, attempt: int) -> None:
        """Reconnect with exponential backoff."""
        delay = min(2 ** attempt, 30)
        logger.warning(
            "MCP stdio reconnecting: %s (attempt %d, backoff %.1fs)",
            self._server_name, attempt + 1, delay,
        )
        await self.disconnect()
        await asyncio.sleep(delay)
        await self.connect()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    async def _send_receive(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request via stdin and read response from stdout."""
        import json

        if not self._connected or self._process is None:
            await self.connect()

        assert self._process is not None
        assert self._process.stdin is not None
        assert self._process.stdout is not None

        payload = json.dumps(request) + "\n"
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

        line = await asyncio.wait_for(
            self._process.stdout.readline(),
            timeout=self._tool_timeout,
        )
        if not line:
            raise ConnectionError(f"MCP stdio EOF from {self._server_name}")
        resp = json.loads(line)
        if "error" in resp:
            err = resp["error"]
            code = err.get("code", -1)
            message = err.get("message", "Unknown JSON-RPC error")
            raise RuntimeError(f"JSON-RPC error {code}: {message}")
        return resp

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC 2.0 notification (no id, no response expected)."""
        import json

        if self._process is None or self._process.stdin is None:
            return
        msg = build_jsonrpc_request(method, params)
        payload = json.dumps(msg) + "\n"
        self._process.stdin.write(payload.encode())
        await self._process.stdin.drain()

    async def initialize(self) -> dict[str, Any]:
        """Perform MCP initialize handshake (JSON-RPC 2.0)."""
        if self._initialized:
            return {"serverInfo": self._server_info or {}, "capabilities": self._server_capabilities or {}}
        if not self._connected or self._process is None:
            await self.connect()
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
            resp = await self._send_receive(req)
            result = resp.get("result", {})
            self._server_info = result.get("serverInfo", {})
            self._server_capabilities = result.get("capabilities", {})
            self._initialized = True
            await self._send_notification("notifications/initialized")
            logger.info(
                "MCP stdio initialized: %s (server=%s)",
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

        for attempt in range(self._max_retries + 1):
            try:
                req = build_jsonrpc_request("tools/list", params={}, request_id=self._next_id())
                resp = await self._send_receive(req)
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
            except (ConnectionError, asyncio.TimeoutError, OSError):
                if attempt < self._max_retries:
                    await self._reconnect(attempt)
                else:
                    raise
        return []

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
                resp = await self._send_receive(req)
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
            except (ConnectionError, asyncio.TimeoutError, OSError) as exc:
                if attempt < self._max_retries:
                    await self._reconnect(attempt)
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
        raise NotImplementedError("Prompts not supported over stdio transport")

    async def list_resources(self) -> list[MCPResourceSpec]:
        return []

    async def read_resource(self, uri: str) -> MCPResourceContent:
        raise NotImplementedError("Resources not supported over stdio transport")
