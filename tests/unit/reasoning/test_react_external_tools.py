"""Tests for ReAct external MCP tool discovery and routing."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from mcp.schemas import MCPToolCallResult, MCPToolSpec
from reasoning.react import ReActReasoningAgent
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry


def _make_agent(mcp_registry=None) -> ReActReasoningAgent:
    llm = AsyncMock()
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    return ReActReasoningAgent(
        llm_adapter=llm,
        tool_registry=registry,
        tool_executor=executor,
        mcp_client_registry=mcp_registry,
    )


class TestDiscoverExternalTools:
    @pytest.mark.asyncio
    async def test_no_registry_returns_empty(self):
        agent = _make_agent(mcp_registry=None)
        names, descs = await agent._discover_external_tools()
        assert names == []
        assert descs == []

    @pytest.mark.asyncio
    async def test_discovers_enabled_tools(self):
        mock_registry = AsyncMock()
        mock_registry.discover_tools = AsyncMock(return_value=[
            MCPToolSpec(
                name="search_scholarly",
                description="Search scholarly papers",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                source="external",
                server_name="scholarly",
                enabled=True,
                tags=["search"],
            ),
            MCPToolSpec(
                name="disabled_tool",
                description="Disabled",
                input_schema={},
                source="external",
                server_name="scholarly",
                enabled=False,
                tags=[],
            ),
        ])
        agent = _make_agent(mcp_registry=mock_registry)
        names, descs = await agent._discover_external_tools()
        assert names == ["search_scholarly"]
        assert len(descs) == 1
        assert descs[0]["source"] == "external"

    @pytest.mark.asyncio
    async def test_tolerates_discovery_failure(self):
        mock_registry = AsyncMock()
        mock_registry.discover_tools = AsyncMock(side_effect=RuntimeError("network error"))
        agent = _make_agent(mcp_registry=mock_registry)
        names, descs = await agent._discover_external_tools()
        assert names == []
        assert descs == []


class TestExecuteTool:
    @pytest.mark.asyncio
    async def test_routes_external_tool_to_mcp(self):
        mock_registry = AsyncMock()
        mock_registry.call_tool = AsyncMock(return_value=MCPToolCallResult(
            call_id="call_ext_1",
            tool_name="search_scholarly",
            status="succeeded",
            output='{"papers": []}',
            server_name="scholarly",
        ))
        agent = _make_agent(mcp_registry=mock_registry)
        result = await agent._execute_tool(
            tool_name="search_scholarly",
            tool_input={"query": "VLN"},
            external_tool_names=["search_scholarly"],
        )
        assert result.status == "succeeded"
        assert result.tool_name == "search_scholarly"
        mock_registry.call_tool.assert_awaited_once_with(
            tool_name="search_scholarly",
            arguments={"query": "VLN"},
        )

    @pytest.mark.asyncio
    async def test_routes_local_tool_to_executor(self):
        agent = _make_agent(mcp_registry=None)
        agent.tool_executor.execute_tool_call = AsyncMock(return_value=AsyncMock(
            status="not_found",
            tool_name="hybrid_retrieve",
        ))
        result = await agent._execute_tool(
            tool_name="hybrid_retrieve",
            tool_input={"query": "test"},
            external_tool_names=["search_scholarly"],
        )
        agent.tool_executor.execute_tool_call.assert_awaited_once()
