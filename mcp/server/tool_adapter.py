from __future__ import annotations

from typing import Any

from mcp.mapping import map_tool_specs_to_mcp_tools
from mcp.schemas import MCPToolCallResult, MCPToolSpec
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry


class MCPToolAdapter:
    def __init__(self, registry: ToolRegistry, executor: ToolExecutor, server_name: str = "local") -> None:
        self.registry = registry
        self.executor = executor
        self.server_name = server_name

    def list_tools(
        self,
        tags: list[str] | None = None,
        names: list[str] | None = None,
        include_disabled: bool = False,
        skill_context: dict[str, Any] | None = None,
    ) -> list[MCPToolSpec]:
        tool_specs = self.registry.filter_tools(
            tags=tags,
            enabled_only=not include_disabled,
            names=names,
            skill_context=skill_context,
        )
        return map_tool_specs_to_mcp_tools(
            tool_specs,
            source="local",
            server_name=self.server_name,
        )

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> MCPToolCallResult:
        result = await self.executor.execute_tool_call(
            tool_name=tool_name,
            tool_input=arguments or {},
            call_id=call_id,
        )
        return MCPToolCallResult(
            call_id=result.call_id,
            tool_name=result.tool_name,
            status=self._map_status(result.status),
            output=result.output,
            error_message=result.error_message,
            latency_ms=result.trace.latency_ms,
            server_name=self.server_name,
            metadata={"source": "local_tool_executor"},
        )

    def _map_status(self, status: str) -> str:
        if status == "validation_error":
            return "invalid_input"
        if status in {"not_found", "disabled", "failed", "succeeded"}:
            return status
        return "failed"
