from types import SimpleNamespace

import pytest

from apps.api.routers.mcp import MCPToolCallRequest, call_tool, discovery, list_tools
from mcp.schemas import MCPPromptSpec, MCPResourceSpec, MCPServerDescriptor, MCPToolCallResult, MCPToolSpec


class MCPServerStub:
    descriptor = MCPServerDescriptor(name="local-mcp", description="Local MCP", enabled=True, metadata={})

    def list_tools(self, **kwargs):
        del kwargs
        return [
            MCPToolSpec(
                name="local_file",
                description="Local file",
                input_schema={"type": "object"},
                enabled=True,
                source="local",
                server_name="local",
            )
        ]

    def list_prompts(self, **kwargs):
        del kwargs
        return [
            MCPPromptSpec(
                name="local_prompt",
                description="Local prompt",
                prompt_key="local.prompt",
                path="local.md",
            )
        ]

    def list_resources(self, **kwargs):
        del kwargs
        return [
            MCPResourceSpec(
                uri="resource://local",
                name="Local resource",
                description="Local resource",
            )
        ]

    async def call_tool(self, **kwargs):
        return MCPToolCallResult(
            call_id=str(kwargs.get("call_id") or "call_local"),
            tool_name=str(kwargs.get("tool_name") or "local_file"),
            status="succeeded",
            output={"ok": True},
            server_name="local",
        )


class MCPClientRegistryStub:
    async def discover_tools(self):
        return [
            MCPToolSpec(
                name="zotero_search_items",
                description="Search local Zotero items",
                input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                output_schema={"type": "object"},
                tags=["zotero", "mcp", "local"],
                enabled=True,
                source="external",
                server_name="zotero-local",
            )
        ]

    async def discover_prompts(self):
        return [
            MCPPromptSpec(
                name="external_prompt",
                description="External prompt",
                prompt_key="external.prompt",
                path="external.md",
            )
        ]

    async def discover_resources(self):
        return [
            MCPResourceSpec(
                uri="resource://external",
                name="External resource",
                description="External resource",
            )
        ]

    async def call_tool(self, *, tool_name, arguments=None, server_name=None, call_id=None):
        del arguments, server_name
        return MCPToolCallResult(
            call_id=str(call_id or "call_external"),
            tool_name=tool_name,
            status="succeeded",
            output={"items": []},
            server_name="zotero-local",
        )


class SkillRegistryStub:
    def allowed_external_mcp_tools(self, task_type: str, preferred_skill_name: str | None = None):
        del task_type, preferred_skill_name
        return {"zotero_search_items"}


class GraphRuntimeStub:
    def __init__(self) -> None:
        self.tool_registry = SimpleNamespace(get_tool=lambda name, include_disabled=False: None)
        self.skill_registry = SkillRegistryStub()
        self.mcp_client_registry = MCPClientRegistryStub()

    def resolve_skill_context(self, *, task_type: str, preferred_skill_name: str | None = None):
        del task_type
        if not preferred_skill_name:
            return None
        return {"name": preferred_skill_name, "preferred_tools": []}

    async def list_external_function_tools(
        self,
        *,
        task_type: str = "function_call",
        preferred_skill_name: str | None = None,
        skill_context=None,
        include_disabled: bool = False,
    ):
        del task_type, preferred_skill_name, skill_context, include_disabled
        return [
            {
                "type": "function",
                "function": {
                    "name": "zotero_search_items",
                    "description": "Search local Zotero items",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
                "server_name": "zotero-local",
                "source": "external",
                "enabled": True,
                "tags": ["zotero", "mcp", "local"],
                "output_schema": {"type": "object"},
            }
        ]


@pytest.mark.asyncio
async def test_list_tools_includes_external_mcp_tools() -> None:
    tools = await list_tools(
        skill_name="research_report",
        mcp_server=MCPServerStub(),
        graph_runtime=GraphRuntimeStub(),
        _=None,
    )

    tool_names = [tool.name for tool in tools]
    assert "local_file" in tool_names
    assert "zotero_search_items" in tool_names
    zotero_tool = next(tool for tool in tools if tool.name == "zotero_search_items")
    assert zotero_tool.source == "external"
    assert zotero_tool.server_name == "zotero-local"


@pytest.mark.asyncio
async def test_discovery_includes_external_mcp_capabilities() -> None:
    snapshot = await discovery(
        mcp_server=MCPServerStub(),
        graph_runtime=GraphRuntimeStub(),
        _=None,
    )

    assert {tool.name for tool in snapshot.tools} >= {"local_file", "zotero_search_items"}
    assert "external_prompt" in {prompt.name for prompt in snapshot.prompts}
    assert "resource://external" in {resource.uri for resource in snapshot.resources}


@pytest.mark.asyncio
async def test_call_tool_routes_to_external_registry_when_local_tool_missing() -> None:
    result = await call_tool(
        MCPToolCallRequest(
            tool_name="zotero_search_items",
            arguments={"query": "rag"},
            skill_name="research_report",
        ),
        graph_runtime=GraphRuntimeStub(),
        mcp_server=MCPServerStub(),
        _=None,
    )

    assert result.status == "succeeded"
    assert result.server_name == "zotero-local"
    assert result.tool_name == "zotero_search_items"
