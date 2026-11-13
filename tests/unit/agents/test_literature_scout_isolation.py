from unittest.mock import AsyncMock

import pytest

from agents.research_supervisor_agent import ResearchSupervisorDecision
from domain.schemas.research import ResearchAgentRunRequest, ResearchReport, ResearchTaskResponse
from runtime.research.supervisor_graph_runtime import ResearchSupervisorGraphRuntime
from tests.unit.services.test_research_supervisor_graph_runtime import build_service, GraphRuntimeStub


def _complete_english_report(task):
    paragraph = "The available evidence compares planning methods, evaluation assumptions, and reproducibility across the selected research papers. " * 7
    report = ResearchReport(
        report_id="english-review", task_id=task.task_id, topic=task.topic,
        generated_at=task.created_at,
        markdown=f"## Background\n{paragraph}\n## Core Problems\n{paragraph}\n## Method Comparison\n{paragraph}\n## Key Findings\n{paragraph} [P1]",
    )
    return report


@pytest.mark.asyncio
@pytest.mark.parametrize("existing", [False, True])
async def test_search_returns_papers_without_invoking_writer_and_preserves_workspace(tmp_path, existing):
    service = build_service(tmp_path)
    runtime = ResearchSupervisorGraphRuntime(research_service=service)
    context = await runtime._build_tool_context(
        request=ResearchAgentRunRequest(message="UAV path planning", mode="research", max_papers=3),
        graph_runtime=GraphRuntimeStub(),
    )
    previous_report = None
    if existing:
        from domain.schemas.research import CreateResearchTaskRequest
        response = await service.create_task(CreateResearchTaskRequest(topic="UAV path planning", run_immediately=False))
        task = response.task
        task.workspace.hypotheses = ["Keep this hypothesis"]
        task.workspace.key_findings = ["Keep this finding"]
        task.workspace.evidence_gaps = ["Keep this gap"]
        task.workspace.metadata["custom"] = "preserve"
        previous_report = ResearchReport(report_id="existing", task_id=task.task_id, topic=task.topic,
                                         generated_at=task.created_at, markdown="Existing report including QA and TODO notes")
        task.report_id = previous_report.report_id
        context.task_response = ResearchTaskResponse(task=task, report=previous_report)
        service.report_service.save_report(previous_report)
    runtime.research_writer_agent.synthesize_async = AsyncMock(side_effect=AssertionError("Scout called writer"))
    runtime.research_writer_agent.plan_todos_async = AsyncMock(side_effect=AssertionError("Scout called writer"))
    result = await runtime.literature_scout_agent.run_action(
        context, ResearchSupervisorDecision(action_name="search_literature", thought="Discover", rationale="Find evidence"),
    )
    assert result.status == "succeeded"
    assert result.state_delta.papers
    assert result.state_delta.report == previous_report
    runtime.research_writer_agent.synthesize_async.assert_not_called()
    runtime.research_writer_agent.plan_todos_async.assert_not_called()
    assert not hasattr(runtime.literature_scout_agent, "research_writer_agent")
    if existing:
        workspace = result.state_delta.task.workspace
        assert workspace.hypotheses == ["Keep this hypothesis"]
        assert "Keep this finding" in workspace.key_findings
        assert workspace.evidence_gaps == ["Keep this gap"]
        assert workspace.metadata["custom"] == "preserve"


@pytest.mark.asyncio
async def test_discovery_workflow_dispatches_writer_after_report_free_search(tmp_path):
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))
    search = runtime.literature_scout_agent.run_action
    writer = runtime.research_writer_agent.run_action
    async def complete_english_review(state):
        from types import SimpleNamespace
        return _complete_english_report(SimpleNamespace(task_id=state.task_id, topic=state.topic, created_at="2026-01-01T00:00:00Z"))
    runtime.research_writer_agent.synthesize_async = AsyncMock(side_effect=complete_english_review)
    async def check_search(context, decision, *, task_type):
        result = await search(context, decision, task_type=task_type)
        assert result.state_delta.report is None
        assert result.state_delta.task.todo_items == []
        return result
    runtime.literature_scout_agent.run_action = check_search
    runtime.research_writer_agent.run_action = AsyncMock(wraps=writer)
    response = await runtime.run(ResearchAgentRunRequest(
        message="UAV path planning", mode="research", max_papers=3,
        sources=["arxiv", "openalex"],
        metadata={"workflow_constraint": "discovery_only"},
    ), graph_runtime=GraphRuntimeStub())
    assert [step.action_name for step in response.trace] == ["search_literature", "write_review", "finalize"]
    runtime.research_writer_agent.run_action.assert_awaited_once()
    assert response.report is not None
    assert response.task.todo_items
    assert response.status == "succeeded"
    assert response.trace[1].status == "succeeded"
    assert response.report.metadata["write_review_retry_count"] == 0


@pytest.mark.asyncio
async def test_refinement_writer_preserves_report_history_and_refreshes_workspace(tmp_path):
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))
    context = await runtime._build_tool_context(
        request=ResearchAgentRunRequest(message="UAV path planning", mode="research", max_papers=3),
        graph_runtime=GraphRuntimeStub(),
    )
    search_decision = ResearchSupervisorDecision(action_name="search_literature", thought="Discover", rationale="Find evidence")
    write_decision = ResearchSupervisorDecision(action_name="write_review", thought="Write", rationale="Summarize evidence")
    search = await runtime.literature_scout_agent.run_action(context, search_decision)
    context.task_response = search.state_delta.task_response
    initial = await runtime.research_writer_agent.run_action(context, write_decision)
    context.task_response = initial.state_delta.task_response
    previous = context.report
    previous.markdown += "\n\n## 研究集合问答补充\nRetained QA answer\n\n## TODO 执行记录\nRetained TODO execution"
    previous.highlights.extend(["问答补充：retained QA", "TODO执行：retained TODO"])
    previous.gaps.append("Historical evidence gap")
    previous.metadata["custom_history"] = {"answer": "retained"}
    context.task.workspace.hypotheses = ["Custom hypothesis"]
    context.task.workspace.research_questions = ["Custom research question"]
    context.task.workspace.metadata["custom_workspace"] = True
    original_id = previous.report_id
    refined = await runtime.literature_scout_agent.run_action(context, search_decision)
    context.task_response = refined.state_delta.task_response
    assert context.report == previous
    context.task_response.warnings = ["OpenAlex search unavailable"]
    generated = previous.model_copy(update={
        "report_id": "new-candidate-id", "markdown": previous.markdown.split("## 研究集合问答补充")[0],
        "highlights": ["Fresh finding"], "gaps": ["Fresh gap"], "metadata": {"new_metadata": True},
    })
    runtime.research_writer_agent.synthesize_async = AsyncMock(return_value=generated)
    rewritten = await runtime.research_writer_agent.run_action(context, write_decision)
    assert runtime.research_writer_agent.synthesize_async.call_args.args[0].warnings == ["OpenAlex search unavailable"]
    report = rewritten.state_delta.report
    workspace = rewritten.state_delta.task.workspace
    assert report.report_id == original_id
    assert report.metadata["custom_history"] == {"answer": "retained"}
    assert report.metadata["new_metadata"] is True
    assert "Retained QA answer" in report.markdown
    assert "Retained TODO execution" in report.markdown
    assert "问答补充：retained QA" in report.highlights
    assert "TODO执行：retained TODO" in report.highlights
    assert "Historical evidence gap" in report.gaps
    assert workspace.key_findings[0] == "Fresh finding"
    assert "Fresh gap" in workspace.evidence_gaps
    assert "Historical evidence gap" in workspace.evidence_gaps
    assert workspace.current_stage != "discover"
    assert workspace.hypotheses == ["Custom hypothesis"]
    assert "Custom research question" in workspace.research_questions
    assert workspace.metadata["custom_workspace"] is True
    assert f"open_todos={sum(item.status == 'open' for item in rewritten.state_delta.task.todo_items)}" in workspace.status_summary
    assert report.workspace == workspace


@pytest.mark.asyncio
async def test_writer_accepts_complete_english_review_without_retry(tmp_path):
    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))
    context = await runtime._build_tool_context(
        request=ResearchAgentRunRequest(message="UAV path planning", mode="research", max_papers=3),
        graph_runtime=GraphRuntimeStub(),
    )
    search = await runtime.literature_scout_agent.run_action(
        context, ResearchSupervisorDecision(action_name="search_literature", thought="Discover", rationale="Find evidence"),
    )
    context.task_response = search.state_delta.task_response
    report = _complete_english_report(context.task)
    runtime.research_writer_agent.synthesize_async = AsyncMock(return_value=report)
    result = await runtime.research_writer_agent.run_action(
        context, ResearchSupervisorDecision(action_name="write_review", thought="Write", rationale="Summarize evidence"),
    )
    assert result.status == "succeeded"
    assert result.state_delta.report.metadata["write_review_retry_count"] == 0
    runtime.research_writer_agent.synthesize_async.assert_awaited_once()
    assert runtime.research_writer_agent._quality_metrics(report.model_copy(update={"markdown": "## Background\n## Method Comparison\n## Key Findings\n[P1]"}))["passed"] is False
    assert runtime.research_writer_agent._quality_metrics(report.model_copy(update={"markdown": report.markdown.replace("[P1]", "")}))["passed"] is False


@pytest.mark.asyncio
async def test_review_handoff_preserves_saved_language_for_english_topic(tmp_path):
    from types import SimpleNamespace

    runtime = ResearchSupervisorGraphRuntime(research_service=build_service(tmp_path))
    context = await runtime._build_tool_context(
        request=ResearchAgentRunRequest(message="UAV path planning", mode="research", max_papers=3),
        graph_runtime=GraphRuntimeStub(),
    )
    search = await runtime.literature_scout_agent.run_action(
        context, ResearchSupervisorDecision(action_name="search_literature", thought="Discover", rationale="Find evidence"),
    )
    context.task_response = search.state_delta.task_response
    context.execution_context = SimpleNamespace(preference_context={"answer_language": "zh-CN"})
    result = await runtime.research_writer_agent.run_action(
        context, ResearchSupervisorDecision(action_name="write_review", thought="Write", rationale="Summarize evidence"),
    )
    assert result.state_delta.report.metadata["answer_language"] == "zh-CN"
    assert "## 研究背景" in result.state_delta.report.markdown
    assert result.status == "succeeded"
