from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from runtime.research.unified_runtime import (
    ACTION_AGENT_CATALOG, build_phase1_unified_blueprint, build_phase1_unified_agent_registry,
)


EXPECTED_ACTIONS = {
    "LiteratureScoutAgent": {"search_literature", "recommend_from_preferences"},
    "ResearchKnowledgeAgent": {
        "import_papers", "sync_to_zotero", "compress_context", "understand_document",
    },
    "ResearchAnalystAgent": {
        "answer_question", "general_answer", "analyze_papers",
        "supervisor_understand_chart", "analyze_paper_figures",
    },
    "ResearchWriterAgent": {"write_review"},
}


def test_four_workers_own_all_existing_actions_once():
    blueprint = build_phase1_unified_blueprint(
        graph_runtime=SimpleNamespace(), research_service=SimpleNamespace(),
    )
    workers = {d.name: set(d.supported_task_types) for d in blueprint.agent_descriptors
               if d.kind == "specialist"}
    assert workers == EXPECTED_ACTIONS
    assert ACTION_AGENT_CATALOG == {
        action: worker for worker, actions in EXPECTED_ACTIONS.items() for action in actions
    }
    assert len(blueprint.agent_descriptors) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("action,owner", sorted(ACTION_AGENT_CATALOG.items()))
async def test_registry_dispatches_each_action_to_its_owned_workflow(action, owner):
    from agents.literature_scout_agent import LiteratureScoutAgent
    from agents.research_analyst_agent import ResearchAnalystAgent
    from agents.research_knowledge_agent import ResearchKnowledgeAgent
    from agents.research_writer_agent import ResearchWriterAgent
    from runtime.research.agent_protocol.base import ResearchStateDelta, ResearchToolResult

    scout = LiteratureScoutAgent(preference_workflow=SimpleNamespace(execute=AsyncMock()))
    knowledge = ResearchKnowledgeAgent()
    analyst = ResearchAnalystAgent()
    writer = ResearchWriterAgent()
    delegates = {worker.name: worker for worker in (scout, knowledge, analyst, writer)}
    # Spy only at domain execution boundaries; exercise the real registry and worker dispatch.
    targets = {
        "search_literature": (scout, "_run_search"),
        "recommend_from_preferences": (scout.preference_workflow, "execute"),
        "import_papers": (knowledge, "_run_import_papers"),
        "sync_to_zotero": (knowledge, "_run_sync_to_zotero"),
        "compress_context": (knowledge, "_run_compress_context"),
        "understand_document": (knowledge.document_workflow, "execute"),
        "answer_question": (analyst, "_run_question"),
        "general_answer": (analyst.general_answer_workflow, "execute"),
        "analyze_papers": (analyst.paper_analysis_workflow, "execute"),
        "supervisor_understand_chart": (analyst.chart_workflow, "execute"),
        "analyze_paper_figures": (analyst.chart_workflow, "execute"),
        "write_review": (writer, "run_action"),
    }
    expected = ResearchToolResult(
        status="partial", observation="domain result", metadata={"warnings": ["provider unavailable"]},
        state_delta=ResearchStateDelta(document_attempted=True),
    )
    spies = {}
    for target, method in targets.values():
        key = (id(target), method)
        if key not in spies:
            spies[key] = AsyncMock(return_value=expected)
            setattr(target, method, spies[key])
    registry = build_phase1_unified_agent_registry(
        graph_runtime=SimpleNamespace(), research_service=SimpleNamespace(), agent_delegates=delegates,
    )
    assert {executor.descriptor.name for executor in registry.list()} == set(EXPECTED_ACTIONS)
    executor = registry.resolve_for_action(action)
    assert executor.descriptor.name == owner
    context, decision = object(), object()
    result = await executor.execute_action(context, decision, action_name=action)
    assert result is expected  # status, warnings and pending state delta survive consolidation
    target, method = targets[action]
    called = spies[(id(target), method)]
    called.assert_awaited_once()
    if action in {"supervisor_understand_chart", "analyze_paper_figures"}:
        called.assert_awaited_once_with(context, decision, task_type=action)
    assert sum(spy.await_count for spy in spies.values()) == 1
    foreign_action = next(name for name, worker in ACTION_AGENT_CATALOG.items() if worker != owner)
    with pytest.raises(ValueError, match="cannot execute"):
        await executor.execute_action(context, decision, action_name=foreign_action)


def test_internal_workflows_have_no_agent_entrypoint_or_identity():
    from tools.research.workflows.chart_analysis import ChartAnalysisWorkflow
    from tools.research.workflows.general_answer import GeneralAnswerWorkflow
    from tools.research.workflows.paper_analysis import PaperAnalysisWorkflow
    from tools.research.workflows.preference_memory import PreferenceRecommendationWorkflow
    from tools.research.workflows.research_document import DocumentWorkflow

    for workflow in (ChartAnalysisWorkflow, GeneralAnswerWorkflow, PaperAnalysisWorkflow,
                     PreferenceRecommendationWorkflow, DocumentWorkflow):
        assert not hasattr(workflow, "run_action")
        assert not hasattr(workflow, "name")
