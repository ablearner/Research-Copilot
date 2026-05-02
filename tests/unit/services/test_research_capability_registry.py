from __future__ import annotations

from types import SimpleNamespace

from mcp.client.registry import MCPClientRegistry
from tools.research.capability_registry import ResearchCapabilityRegistry
from tooling.registry import ToolRegistry
from tooling.research_supervisor_tool_specs import build_research_supervisor_tool_spec
from tooling.schemas import ToolSpec
from pydantic import BaseModel


class _EmptyModel(BaseModel):
    pass


async def _noop_handler(**_: object) -> dict[str, str]:
    return {"status": "ok"}


def test_research_capability_registry_summarizes_action_runtime_and_mcp_layers() -> None:
    unified_registry = ToolRegistry()
    unified_registry.register(
        build_research_supervisor_tool_spec("search_literature", _noop_handler),
    )
    runtime_registry = ToolRegistry()
    runtime_registry.register(
        ToolSpec(
            name="hybrid_retrieve",
            description="Retrieve evidence",
            input_schema=_EmptyModel,
            output_schema=_EmptyModel,
            handler=_noop_handler,
            tags=["retrieval", "search"],
        )
    )
    runtime_registry.register(
        ToolSpec(
            name="code_execution",
            description="Execute local code",
            input_schema=_EmptyModel,
            output_schema=_EmptyModel,
            handler=_noop_handler,
            tags=["execution"],
        )
    )
    mcp_registry = MCPClientRegistry()
    mcp_registry.register_server("zotero", SimpleNamespace())

    runtime = SimpleNamespace(tool_registry=unified_registry)
    graph_runtime = SimpleNamespace(
        tool_registry=runtime_registry,
        mcp_client_registry=mcp_registry,
    )

    registry = ResearchCapabilityRegistry(runtime=runtime)
    summary = registry.inventory_summary(graph_runtime=graph_runtime)

    assert summary["action_count"] == 1
    assert summary["knowledge_count"] == 1
    assert summary["runtime_count"] == 1
    assert summary["mcp_server_count"] == 1
    assert summary["action_names"] == ["search_literature"]
    assert summary["knowledge_names"] == ["hybrid_retrieve"]
    assert summary["runtime_names"] == ["code_execution"]
    assert summary["mcp_server_names"] == ["zotero"]
