from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from tools.research.external_tool_gateway import ResearchExternalToolGateway


CapabilityKind = Literal["action", "knowledge", "runtime", "mcp_server"]


@dataclass(frozen=True, slots=True)
class ResearchCapabilityDescriptor:
    name: str
    kind: CapabilityKind
    enabled: bool
    source_registry: str
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None


class ResearchCapabilityRegistry:
    """Read-only unified capability inventory over supervisor, runtime, and MCP surfaces."""

    def __init__(self, *, runtime: Any) -> None:
        self.runtime = runtime

    def list_capabilities(
        self,
        *,
        graph_runtime: Any | None = None,
        include_disabled: bool = False,
    ) -> list[ResearchCapabilityDescriptor]:
        capabilities: list[ResearchCapabilityDescriptor] = []
        unified_registry = getattr(self.runtime, "tool_registry", None)
        if unified_registry is not None:
            for tool in unified_registry.list_tools(include_disabled=include_disabled):
                if tool.toolset == "supervisor_action":
                    capabilities.append(
                        ResearchCapabilityDescriptor(
                            name=tool.name,
                            kind="action",
                            enabled=bool(tool.enabled),
                            source_registry="unified_tool_registry",
                            tags=tuple(tool.tags),
                            metadata={"category": tool.category},
                        )
                    )
        runtime_registry = getattr(graph_runtime, "tool_registry", None)
        if runtime_registry is not None:
            for tool in runtime_registry.list_tools(include_disabled=include_disabled):
                capabilities.append(
                    ResearchCapabilityDescriptor(
                        name=tool.name,
                        kind=self._runtime_kind(tool),
                        enabled=bool(tool.enabled),
                        source_registry="runtime_tool_registry",
                        tags=tuple(tool.tags),
                        metadata={"category": tool.category},
                    )
                )
        external_gateway = ResearchExternalToolGateway(graph_runtime=graph_runtime)
        if external_gateway.is_configured():
            for server_name in external_gateway.list_servers():
                capabilities.append(
                    ResearchCapabilityDescriptor(
                        name=server_name,
                        kind="mcp_server",
                        enabled=True,
                        source_registry="research_external_tool_gateway",
                        metadata={"server_name": server_name},
                    )
                )
        return capabilities

    def local_capability_names(
        self,
        *,
        graph_runtime: Any | None = None,
        include_disabled: bool = False,
    ) -> list[str]:
        return sorted(
            {
                item.name
                for item in self.list_capabilities(
                    graph_runtime=graph_runtime,
                    include_disabled=include_disabled,
                )
                if item.kind != "mcp_server"
            }
        )

    def inventory_summary(
        self,
        *,
        graph_runtime: Any | None = None,
    ) -> dict[str, Any]:
        capabilities = self.list_capabilities(
            graph_runtime=graph_runtime,
            include_disabled=False,
        )
        action_names = sorted(item.name for item in capabilities if item.kind == "action")
        knowledge_names = sorted(item.name for item in capabilities if item.kind == "knowledge")
        runtime_names = sorted(item.name for item in capabilities if item.kind == "runtime")
        mcp_server_names = sorted(item.name for item in capabilities if item.kind == "mcp_server")
        return {
            "local_capability_count": len(action_names) + len(knowledge_names) + len(runtime_names),
            "action_count": len(action_names),
            "knowledge_count": len(knowledge_names),
            "runtime_count": len(runtime_names),
            "mcp_server_count": len(mcp_server_names),
            "action_names": action_names,
            "knowledge_names": knowledge_names,
            "runtime_names": runtime_names,
            "mcp_server_names": mcp_server_names,
        }

    def _runtime_kind(self, tool_spec: Any) -> CapabilityKind:
        tags = set(getattr(tool_spec, "tags", []) or [])
        if {
            "retrieval",
            "graph",
            "summary",
            "answer",
            "document",
            "chart",
            "parse",
            "index",
            "vision",
            "search",
        } & tags:
            return "knowledge"
        return "runtime"
