from __future__ import annotations

import logging
from typing import Any

from mcp.mapping import map_tool_specs_to_mcp_tools
from tooling.schemas import ToolSpec
from tooling.serializers import tool_specs_to_openai_functions

logger = logging.getLogger(__name__)


class ToolRegistryError(RuntimeError):
    """Raised when tool registration or lookup fails."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool_spec: ToolSpec, replace: bool = False) -> ToolSpec:
        if tool_spec.name in self._tools and not replace:
            raise ToolRegistryError(f"Tool already registered: {tool_spec.name}")
        self._tools[tool_spec.name] = tool_spec
        logger.info(
            "Tool registered",
            extra={
                "tool_name": tool_spec.name,
                "enabled": tool_spec.enabled,
                "tags": tool_spec.tags,
            },
        )
        return tool_spec

    def register_many(self, tool_specs: list[ToolSpec], replace: bool = False) -> None:
        for tool_spec in tool_specs:
            self.register(tool_spec, replace=replace)

    def get_tool(self, name: str, include_disabled: bool = False) -> ToolSpec | None:
        tool_spec = self._tools.get(name)
        if tool_spec is None:
            return None
        if not include_disabled and not tool_spec.enabled:
            return None
        return tool_spec

    def list_tools(self, include_disabled: bool = True) -> list[ToolSpec]:
        tools = list(self._tools.values())
        if include_disabled:
            return tools
        return [tool for tool in tools if tool.enabled]

    def filter_tools(
        self,
        tags: list[str] | None = None,
        enabled_only: bool = True,
        names: list[str] | None = None,
        skill_context: dict[str, Any] | Any | None = None,
    ) -> list[ToolSpec]:
        normalized_tags = set(tags or [])
        normalized_names = set(names or [])
        preferred_tools = set(self._preferred_tools_from_skill_context(skill_context))
        tools = self.list_tools(include_disabled=not enabled_only)
        filtered: list[ToolSpec] = []
        for tool_spec in tools:
            if normalized_names and tool_spec.name not in normalized_names:
                continue
            if preferred_tools and tool_spec.name not in preferred_tools:
                continue
            if normalized_tags and not (normalized_tags & set(tool_spec.tags)):
                continue
            filtered.append(tool_spec)
        return filtered

    def as_openai_function_tools(
        self,
        tags: list[str] | None = None,
        enabled_only: bool = True,
        names: list[str] | None = None,
        skill_context: dict[str, Any] | Any | None = None,
    ) -> list[dict]:
        tool_specs = self.filter_tools(
            tags=tags,
            enabled_only=enabled_only,
            names=names,
            skill_context=skill_context,
        )
        return tool_specs_to_openai_functions(tool_specs)

    def as_mcp_tools(
        self,
        tags: list[str] | None = None,
        enabled_only: bool = True,
        names: list[str] | None = None,
        skill_context: dict[str, Any] | Any | None = None,
        source: str = "local",
        server_name: str | None = None,
    ):
        tool_specs = self.filter_tools(
            tags=tags,
            enabled_only=enabled_only,
            names=names,
            skill_context=skill_context,
        )
        return map_tool_specs_to_mcp_tools(
            tool_specs,
            source=source,
            server_name=server_name,
        )

    def _preferred_tools_from_skill_context(
        self,
        skill_context: dict[str, Any] | Any | None,
    ) -> list[str]:
        if skill_context is None:
            return []
        if isinstance(skill_context, dict):
            preferred = skill_context.get("preferred_tools")
            return preferred if isinstance(preferred, list) else []
        preferred = getattr(skill_context, "preferred_tools", None)
        return preferred if isinstance(preferred, list) else []
