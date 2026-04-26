from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from domain.schemas.research_context import ResearchContext, ResearchContextSlice


@dataclass(slots=True)
class ResearchExecutionContext:
    session_id: str | None = None
    session_context: dict[str, Any] = field(default_factory=dict)
    memory_hints: dict[str, Any] = field(default_factory=dict)
    task_context: dict[str, Any] = field(default_factory=dict)
    preference_context: dict[str, Any] = field(default_factory=dict)
    conversation_context: dict[str, Any] = field(default_factory=dict)
    research_context: ResearchContext | None = None
    context_slices: dict[str, ResearchContextSlice] = field(default_factory=dict)

    @property
    def memory_enabled(self) -> bool:
        return bool(self.session_context.get("memory_enabled"))
