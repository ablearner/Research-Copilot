"""Tests for MCP standardization: JSON-RPC 2.0, initialize handshake, config loading, external tool injection."""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp.client.base import BaseMCPClient, build_jsonrpc_request, _MCP_PROTOCOL_VERSION, _CLIENT_INFO
from mcp.client.registry import MCPClientRegistry
from mcp.client.stdio_client import StdioMCPClient
from mcp.client.http_client import HttpMCPClient
from mcp.schemas import MCPToolCallResult, MCPToolSpec


# ── JSON-RPC 2.0 envelope ──────────────────────────────────────────

class TestBuildJsonrpcRequest:
    def test_basic_request(self):
        req = build_jsonrpc_request("tools/list", params={}, request_id=1)
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "tools/list"
        assert req["id"] == 1
        assert req["params"] == {}

    def test_notification_no_id(self):
        req = build_jsonrpc_request("notifications/initialized")
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "notifications/initialized"
        assert "id" not in req

    def test_initialize_request(self):
        req = build_jsonrpc_request(
            "initialize",
            params={
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": _CLIENT_INFO,
            },
            request_id=0,
        )
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "initialize"
        assert req["id"] == 0
        assert req["params"]["protocolVersion"] == _MCP_PROTOCOL_VERSION
        assert req["params"]["clientInfo"]["name"] == "research-copilot"

    def test_tools_call_request(self):
        req = build_jsonrpc_request(
            "tools/call",
            params={"name": "search_papers", "arguments": {"query": "VLN"}},
            request_id=42,
        )
        assert req["jsonrpc"] == "2.0"
        assert req["id"] == 42
        assert req["params"]["name"] == "search_papers"
        assert req["params"]["arguments"]["query"] == "VLN"


# ── BaseMCPClient defaults ──────────────────────────────────────────

class TestBaseMCPClientDefaults:
    def test_default_initialized_false(self):
        class DummyClient(BaseMCPClient):
            @property
            def server_name(self): return "dummy"
            async def list_tools(self): return []
            async def call_tool(self, **kw): raise NotImplementedError
            async def list_prompts(self): return []
            async def get_prompt(self, **kw): raise NotImplementedError
            async def list_resources(self): return []
            async def read_resource(self, uri): raise NotImplementedError

        client = DummyClient()
        assert client._initialized is False
        assert client._server_info is None

    @pytest.mark.asyncio
    async def test_initialize_returns_empty_dict(self):
        class DummyClient(BaseMCPClient):
            @property
            def server_name(self): return "dummy"
            async def list_tools(self): return []
            async def call_tool(self, **kw): raise NotImplementedError
            async def list_prompts(self): return []
            async def get_prompt(self, **kw): raise NotImplementedError
            async def list_resources(self): return []
            async def read_resource(self, uri): raise NotImplementedError

        client = DummyClient()
        result = await client.initialize()
        assert result == {}


# ── StdioMCPClient JSON-RPC 2.0 ─────────────────────────────────────

class TestStdioMCPClientJsonRpc:
    def test_initial_state(self):
        client = StdioMCPClient(server_name="test", command="/bin/echo")
        assert client._initialized is False
        assert client._request_id == 0

    def test_next_id_increments(self):
        client = StdioMCPClient(server_name="test", command="/bin/echo")
        assert client._next_id() == 1
        assert client._next_id() == 2
        assert client._next_id() == 3


# ── HttpMCPClient JSON-RPC 2.0 ──────────────────────────────────────

class TestHttpMCPClientJsonRpc:
    def test_initial_state(self):
        client = HttpMCPClient(server_name="test", base_url="http://localhost:8080")
        assert client._initialized is False
        assert client._request_id == 0

    def test_next_id_increments(self):
        client = HttpMCPClient(server_name="test", base_url="http://localhost:8080")
        assert client._next_id() == 1
        assert client._next_id() == 2


# ── register_from_json_file ─────────────────────────────────────────

class TestRegisterFromJsonFile:
    def test_file_not_found_returns_empty(self):
        reg = MCPClientRegistry()
        result = reg.register_from_json_file("/nonexistent/path.json")
        assert result == []

    def test_invalid_json_returns_empty(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json {{{", encoding="utf-8")
        reg = MCPClientRegistry()
        result = reg.register_from_json_file(bad_file)
        assert result == []

    def test_flat_format(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "my-server": {
                "transport": "stdio",
                "command": "/bin/true",
            }
        }), encoding="utf-8")
        reg = MCPClientRegistry()
        result = reg.register_from_json_file(cfg)
        assert "my-server" in result
        assert isinstance(reg.get_server("my-server"), StdioMCPClient)

    def test_servers_wrapper_format(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "_comment": "test config",
            "servers": {
                "srv-a": {"transport": "stdio", "command": "/bin/true"},
                "srv-b": {"transport": "http", "url": "http://localhost:3000"},
            }
        }), encoding="utf-8")
        reg = MCPClientRegistry()
        result = reg.register_from_json_file(cfg)
        assert set(result) == {"srv-a", "srv-b"}

    def test_disabled_server_skipped(self, tmp_path):
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({
            "enabled-srv": {"transport": "stdio", "command": "/bin/true", "enabled": True},
            "disabled-srv": {"transport": "stdio", "command": "/bin/true", "enabled": False},
        }), encoding="utf-8")
        reg = MCPClientRegistry()
        result = reg.register_from_json_file(cfg)
        assert "enabled-srv" in result
        assert "disabled-srv" not in result


# ── initialize_all ───────────────────────────────────────────────────

class TestInitializeAll:
    @pytest.mark.asyncio
    async def test_calls_initialize_on_all_clients(self):
        reg = MCPClientRegistry()
        mock1 = AsyncMock(spec=BaseMCPClient)
        mock1.server_name = "s1"
        mock1.initialize = AsyncMock(return_value={})
        mock2 = AsyncMock(spec=BaseMCPClient)
        mock2.server_name = "s2"
        mock2.initialize = AsyncMock(return_value={})
        reg.register_server("s1", mock1)
        reg.register_server("s2", mock2)
        await reg.initialize_all()
        mock1.initialize.assert_awaited_once()
        mock2.initialize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tolerates_initialize_failure(self):
        reg = MCPClientRegistry()
        mock_ok = AsyncMock(spec=BaseMCPClient)
        mock_ok.server_name = "ok"
        mock_ok.initialize = AsyncMock(return_value={})
        mock_fail = AsyncMock(spec=BaseMCPClient)
        mock_fail.server_name = "fail"
        mock_fail.initialize = AsyncMock(side_effect=RuntimeError("boom"))
        reg.register_server("ok", mock_ok)
        reg.register_server("fail", mock_fail)
        await reg.initialize_all()
        mock_ok.initialize.assert_awaited_once()
        mock_fail.initialize.assert_awaited_once()


# ── HttpMCPClient initialize handshake ───────────────────────────────

class TestHttpMCPClientInitialize:
    @pytest.mark.asyncio
    async def test_initialize_sets_server_info(self):
        client = HttpMCPClient(server_name="test-http", base_url="http://localhost:9999")
        mock_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test-server", "version": "0.1"},
            },
        }
        with patch.object(client, "_post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.initialize()
        assert client._initialized is True
        assert client._server_info["name"] == "test-server"
        assert result["serverInfo"]["name"] == "test-server"

    @pytest.mark.asyncio
    async def test_initialize_graceful_on_failure(self):
        client = HttpMCPClient(server_name="fail-http", base_url="http://localhost:9999")
        with patch.object(client, "_post", new_callable=AsyncMock, side_effect=RuntimeError("connect failed")):
            result = await client.initialize()
        assert client._initialized is True
        assert result == {}

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        client = HttpMCPClient(server_name="test-http", base_url="http://localhost:9999")
        client._initialized = True
        client._server_info = {"name": "cached"}
        client._server_capabilities = {"tools": {}}
        result = await client.initialize()
        assert result["serverInfo"]["name"] == "cached"


# ── HttpMCPClient error handling ─────────────────────────────────────

class TestHttpMCPClientErrorHandling:
    @pytest.mark.asyncio
    async def test_post_raises_on_jsonrpc_error(self):
        client = HttpMCPClient(server_name="err", base_url="http://localhost:9999")
        client._initialized = True
        import httpx
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32600, "message": "Invalid Request"},
        }

        async def mock_post(url, json=None):
            return mock_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.post = mock_post

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(RuntimeError, match="JSON-RPC error -32600"):
                await client._post("/mcp", {"method": "test"})
