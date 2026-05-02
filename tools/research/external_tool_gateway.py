from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class ResearchExternalToolGateway:
    """Unified research-domain facade over MCP/external tool calls."""

    def __init__(
        self,
        *,
        graph_runtime: Any | None = None,
        registry: Any | None = None,
    ) -> None:
        self.graph_runtime = graph_runtime
        self.registry = registry

    def bind_runtime(self, graph_runtime: Any | None) -> None:
        self.graph_runtime = graph_runtime

    def is_configured(self) -> bool:
        return self._registry() is not None

    async def call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        server_name: str | None = None,
        call_id: str | None = None,
    ) -> Any:
        registry = self._registry()
        if registry is None:
            return SimpleNamespace(
                status="not_configured",
                output=None,
                error_message="External MCP registry is not configured.",
            )
        kwargs: dict[str, Any] = {
            "tool_name": tool_name,
            "arguments": arguments,
        }
        if server_name is not None:
            kwargs["server_name"] = server_name
        if call_id is not None:
            kwargs["call_id"] = call_id
        return await registry.call_tool(**kwargs)

    async def discover_tools(self) -> list[Any]:
        registry = self._registry()
        if registry is None:
            return []
        discover = getattr(registry, "discover_tools", None)
        if not callable(discover):
            return []
        return await discover()

    def list_servers(self) -> list[str]:
        registry = self._registry()
        if registry is None:
            return []
        list_servers = getattr(registry, "list_servers", None)
        if not callable(list_servers):
            return []
        return list(list_servers())

    def _registry(self) -> Any | None:
        if self.registry is not None:
            return self.registry
        if self.graph_runtime is None:
            return None
        registry = getattr(self.graph_runtime, "external_tool_registry", None)
        if registry is not None:
            return registry
        return getattr(self.graph_runtime, "mcp_client_registry", None)
