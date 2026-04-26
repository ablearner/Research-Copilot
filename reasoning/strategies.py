from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class QueryPlanningStrategy(Protocol):
    llm_adapter: Any | None

    async def plan_queries(
        self,
        *,
        objective: str,
        seed_queries: list[str],
        context: dict[str, Any] | None = None,
        max_queries: int = 4,
        agent_name: str = "planner",
    ) -> Any:
        ...


class AnswerSynthesisStrategy(Protocol):
    llm_adapter: Any | None

    async def reason(
        self,
        *,
        question: str,
        evidence_bundle: Any,
        retrieval_result: Any | None = None,
        metadata: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        preference_context: dict[str, Any] | None = None,
        memory_hints: dict[str, Any] | None = None,
        skill_context: dict[str, Any] | None = None,
    ) -> Any:
        ...


class ToolReasoningStrategy(Protocol):
    llm_adapter: Any | None

    async def reason(
        self,
        question: str,
        available_tool_names: list[str] | None = None,
        max_steps: int = 4,
        metadata: dict[str, Any] | None = None,
        session_context: dict[str, Any] | None = None,
        task_context: dict[str, Any] | None = None,
        preference_context: dict[str, Any] | None = None,
        skill_context: dict[str, Any] | None = None,
        initial_retrieval_result: Any | None = None,
        initial_evidence_bundle: Any | None = None,
    ) -> Any:
        ...


@dataclass(slots=True)
class ReasoningStrategySet:
    """Explicit strategy bundle injected into agents and runtimes."""

    query_planning: QueryPlanningStrategy | None = None
    answer_synthesis: AnswerSynthesisStrategy | None = None
    tool_reasoning: ToolReasoningStrategy | None = None

    @property
    def llm_adapter(self) -> Any | None:
        for strategy in (self.query_planning, self.answer_synthesis, self.tool_reasoning):
            adapter = getattr(strategy, "llm_adapter", None)
            if adapter is not None:
                return adapter
        return None

    @property
    def plan_and_solve_reasoning_agent(self) -> QueryPlanningStrategy | None:
        return self.query_planning

    @property
    def cot_reasoning_agent(self) -> AnswerSynthesisStrategy | None:
        return self.answer_synthesis

    @property
    def react_reasoning_agent(self) -> ToolReasoningStrategy | None:
        return self.tool_reasoning
