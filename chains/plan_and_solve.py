from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError, format_llm_error, is_expected_provider_error
from adapters.llm.langchain_binding import LangChainProviderBinding, ensure_provider_binding

logger = logging.getLogger(__name__)


class PlanAndSolveQueryPlan(BaseModel):
    plan_steps: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    reasoning_summary: str = ""


class PlanAndSolveReasoningAgent:
    """Lightweight planner that decomposes search goals into query plans."""

    def __init__(self, llm_adapter: BaseLLMAdapter | None = None) -> None:
        self.llm_adapter = llm_adapter
        self.binding = self._build_binding(llm_adapter)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a Plan-and-Solve query planner. "
                        "First plan the subproblems privately, then return only structured output. "
                        "Do not reveal private chain of thought. "
                        "Produce a short list of plan_steps, a deduplicated query set, and a concise reasoning_summary."
                    ),
                ),
                ("human", "{payload_json}"),
            ]
        )

    async def plan_queries(
        self,
        *,
        objective: str,
        seed_queries: list[str],
        context: dict[str, Any] | None = None,
        max_queries: int = 4,
        agent_name: str = "planner",
    ) -> PlanAndSolveQueryPlan:
        heuristic = self._fallback_plan(
            objective=objective,
            seed_queries=seed_queries,
            max_queries=max_queries,
        )
        if self.binding is None:
            return heuristic

        payload = {
            "objective": objective,
            "seed_queries": seed_queries,
            "max_queries": max(1, min(max_queries, 6)),
            "context": context or {},
        }
        try:
            messages = await self.prompt.ainvoke(
                {"payload_json": json.dumps(payload, ensure_ascii=False, indent=2)}
            )
            response = await self.binding.ainvoke_structured(
                messages=messages.to_messages(),
                response_model=PlanAndSolveQueryPlan,
                metadata={"agent": agent_name, "task": "plan_and_solve_queries"},
            )
            normalized_queries = self._normalize_queries(
                [*seed_queries, *response.queries],
                limit=max_queries,
            )
            if not normalized_queries:
                return heuristic
            return response.model_copy(update={"queries": normalized_queries})
        except (LLMAdapterError, OSError, ValueError, Exception) as exc:
            if is_expected_provider_error(exc):
                logger.warning(
                    "Plan-and-Solve query planning failed; using heuristic fallback. cause=%s",
                    format_llm_error(exc),
                )
            else:
                logger.warning("Plan-and-Solve query planning failed; using heuristic fallback", exc_info=exc)
            return heuristic

    def _fallback_plan(
        self,
        *,
        objective: str,
        seed_queries: list[str],
        max_queries: int,
    ) -> PlanAndSolveQueryPlan:
        queries = self._normalize_queries(seed_queries or [objective], limit=max_queries)
        return PlanAndSolveQueryPlan(
            plan_steps=[
                "Identify the core research objective or collection question.",
                "Break it into search facets that improve coverage and evidence precision.",
                "Emit a compact set of high-yield queries for the next retrieval round.",
            ],
            queries=queries,
            reasoning_summary="The planner kept the strongest seed queries and organized them into a compact next-step query set.",
        )

    def _normalize_queries(self, queries: list[str], *, limit: int) -> list[str]:
        deduped: list[str] = []
        for query in queries:
            normalized = " ".join(str(query).strip().split())
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped[: max(1, min(limit, 6))]

    def _build_binding(self, llm_adapter: BaseLLMAdapter | None) -> LangChainProviderBinding | None:
        if llm_adapter is None:
            return None
        try:
            return ensure_provider_binding(llm_adapter)
        except TypeError:
            return None
