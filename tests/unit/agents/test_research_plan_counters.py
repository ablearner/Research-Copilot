from types import SimpleNamespace

import pytest

from adapters.llm.base import LLMAdapterError
from agents.research_supervisor_agent import ResearchSupervisorAgent, ResearchSupervisorState
from domain.schemas.research_context import ResearchContextSlice
from domain.schemas.research_plan import PlanStep, ResearchPlan


def failed_plan():
    return ResearchPlan(plan_id="p", steps=[PlanStep(step_id=1, action="search_literature", status="failed")],
                        status="needs_replan", reason="No candidates found")


@pytest.mark.asyncio
async def test_context_compression_does_not_count_as_replanning():
    async def unexpected_call(**kwargs):
        pytest.fail("Compression must run before replanning")
    manager = ResearchSupervisorAgent(llm_adapter=SimpleNamespace(generate_structured=unexpected_call))
    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(goal="research", execution_plan=failed_plan(), planning_llm_calls=1),
        context_slice=ResearchContextSlice(known_conclusions=["x" * 200_000]),
    )
    assert decision.action_name == "compress_context"
    update = decision.metadata["state_update"]
    assert update["planning_llm_calls"] == 1
    assert update["replan_count"] == 0
    assert update["execution_plan"]["replan_attempts"] == 0


@pytest.mark.asyncio
async def test_failed_replanning_call_is_counted_once():
    async def unavailable(**kwargs):
        raise LLMAdapterError("Unavailable")
    manager = ResearchSupervisorAgent(llm_adapter=SimpleNamespace(generate_structured=unavailable))
    decision = await manager.decide_next_action_async(
        ResearchSupervisorState(goal="research", execution_plan=failed_plan(), planning_llm_calls=1),
    )
    assert decision.action_name == "finalize"
    update = decision.metadata["state_update"]
    assert update["planning_llm_calls"] == 2
    assert update["replan_count"] == 1
    assert update["execution_plan"]["replan_attempts"] == 1
