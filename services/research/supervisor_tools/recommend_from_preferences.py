"""Recommend from preferences supervisor tool."""

from __future__ import annotations

from agents.preference_memory_agent import PreferenceMemoryAgent
from agents.research_supervisor_agent import ResearchSupervisorDecision
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.unified_action_adapters import resolve_active_message


class RecommendFromPreferencesTool:
    name = "recommend_from_preferences"

    def __init__(self, *, preference_memory_agent: PreferenceMemoryAgent) -> None:
        self.preference_memory_agent = preference_memory_agent

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        question = str(payload.get("goal") or context.request.message or "").strip()
        top_k = max(
            1,
            min(
                10,
                int(
                    payload.get("top_k")
                    or context.request.recommendation_top_k
                    or 6
                ),
            ),
        )
        days_back = max(
            1,
            int(
                payload.get("days_back")
                or context.request.days_back
                or 30
            ),
        )
        raw_sources = payload.get("sources")
        sources = (
            [str(item).strip().lower() for item in raw_sources if str(item).strip()]
            if isinstance(raw_sources, list)
            else list(context.request.sources)
        )
        recommendation_output = await self.preference_memory_agent.recommend_recent_papers(
            question=question,
            days_back=days_back,
            top_k=top_k,
            sources=sources,
            include_notification=True,
        )
        context.preference_recommendation_result = recommendation_output
        recommendations = list(recommendation_output.recommendations)
        if not recommendations:
            return ResearchToolResult(
                status="skipped",
                observation="no personalized paper recommendations could be generated from long-term preferences",
                metadata={
                    "reason": "no_preference_recommendations",
                    **recommendation_output.model_dump(mode="json"),
                },
            )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"generated {len(recommendations)} personalized recommendations from long-term preferences"
            ),
            metadata=recommendation_output.model_dump(mode="json"),
        )
