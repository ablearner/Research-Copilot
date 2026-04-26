from __future__ import annotations

from tooling.registry import ToolRegistry
from tooling.research_function_specs import RESEARCH_FUNCTION_DEFINITIONS, build_research_tool_spec
from tooling.schemas import ToolHandler, ToolSpec


class ResearchFunctionRegistry:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def register(
        self,
        name: str,
        handler: ToolHandler,
        *,
        replace: bool = False,
        enabled: bool = True,
        max_retries: int = 3,
        tags: list[str] | None = None,
    ) -> ToolSpec:
        spec = build_research_tool_spec(
            name,
            handler,
            enabled=enabled,
            max_retries=max_retries,
            tags=tags,
        )
        return self.registry.register(spec, replace=replace)

    def register_many(
        self,
        handlers: dict[str, ToolHandler],
        *,
        replace: bool = False,
        max_retries: int = 3,
    ) -> None:
        for name, handler in handlers.items():
            if name not in RESEARCH_FUNCTION_DEFINITIONS:
                continue
            self.register(
                name,
                handler,
                replace=replace,
                max_retries=max_retries,
            )

    def list_registered(self) -> list[str]:
        return [tool.name for tool in self.registry.list_tools(include_disabled=True)]

    def get_registry(self) -> ToolRegistry:
        return self.registry
