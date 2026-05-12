from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.research_qa_agent import ResearchQAAgent
from domain.schemas.agent_message import AgentMessage
from domain.schemas.research import (
    ResearchAgentRunRequest,
    ResearchTask,
    ResearchTaskResponse,
    ResearchWorkspaceState,
)
from runtime.research.agent_protocol.base import ResearchAgentToolContext
from tools.research.visual_intent import VisualIntentDecision


class VisualIntentRouterStub:
    def __init__(self, decision: VisualIntentDecision) -> None:
        self.decision = decision
        self.calls: list[dict] = []

    async def decide_async(self, **kwargs):
        self.calls.append(dict(kwargs))
        return self.decision


class ResearchServiceStub:
    def __init__(self, visual_intent_router: VisualIntentRouterStub) -> None:
        self.visual_intent_router = visual_intent_router


@pytest.mark.asyncio
async def test_run_action_replans_new_visual_search_before_reusing_workspace_anchor() -> None:
    router = VisualIntentRouterStub(
        VisualIntentDecision(
            intent="new_visual_search",
            reuse_current_anchor=False,
            search_new_figure=True,
            target_description="实验结果直方图",
            exclude_figure_ids=["paper-x:old"],
            confidence=0.91,
            rationale="The user asked to locate a new result histogram.",
        )
    )
    task = ResearchTask(
        task_id="task_visual_replan",
        topic="visual QA",
        created_at="2026-05-13T00:00:00+00:00",
        updated_at="2026-05-13T00:00:00+00:00",
        workspace=ResearchWorkspaceState(
            metadata={
                "last_visual_anchor": {
                    "figure_id": "paper-x:old",
                    "image_path": "/tmp/old.png",
                    "chart_id": "old",
                },
                "last_visual_anchor_figure": {
                    "figure_id": "paper-x:old",
                    "image_path": "/tmp/old.png",
                    "caption": "Qualitative examples.",
                },
            }
        ),
    )
    active_message = AgentMessage(
        task_id="qa-1",
        agent_from="ResearchSupervisorAgent",
        agent_to="ResearchQAAgent",
        task_type="answer_question",
        payload={
            "question": "给我提供实验结果直方图，并分析",
            "qa_route": "chart_drilldown",
            "routing_authority": "supervisor_llm",
        },
    )
    context = ResearchAgentToolContext(
        request=ResearchAgentRunRequest(
            message="给我提供实验结果直方图，并分析",
            task_id=task.task_id,
        ),
        research_service=ResearchServiceStub(router),
        graph_runtime=object(),
        task_response=ResearchTaskResponse(task=task),
    )

    result = await ResearchQAAgent().run_action(
        context=context,
        decision=SimpleNamespace(metadata={"active_message": active_message}),
    )

    assert result.status == "skipped"
    assert result.metadata["reason"] == "visual_search_requires_figure_selection"
    envelope = result.metadata["observation_envelope"]
    assert envelope["suggested_next_actions"] == ["analyze_paper_figures"]
    assert envelope["state_delta"]["target_description"] == "实验结果直方图"
    assert envelope["state_delta"]["exclude_figure_ids"] == ["paper-x:old"]
    assert router.calls[0]["current_visual_anchor"]["figure_id"] == "paper-x:old"
