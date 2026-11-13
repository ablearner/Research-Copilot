import json
from types import SimpleNamespace

import pytest

from tools.research.workflows.paper_analysis import PaperAnalysisWorkflow
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.research import PaperCandidate, ResearchAgentRunRequest, ResearchReport, ResearchTask, ResearchTaskResponse
from domain.schemas.research_context import ResearchContextSlice
from domain.schemas.worker_task import WorkerTaskInput
from runtime.research.agent_protocol.base import ResearchAgentToolContext
from runtime.research.unified_action_adapters import build_expert_context, build_paper_analysis_input, build_review_draft_input
from tools.research.paper_analysis import PaperAnalysisTool
from tools.research.survey_writing import SurveyWritingTool


class CapturingLLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def generate_structured(self, *, prompt, input_data, response_model):
        self.calls.append(input_data)
        return response_model.model_validate(self.result)


def _papers():
    return [PaperCandidate(paper_id=paper_id, title=paper_id, source="arxiv") for paper_id in ("p1", "p2")]


def _task_context(selected_papers=None):
    return WorkerTaskInput(
        task_id="write-2", action_name="write_review",
        instruction="Use the prior comparison to explain p1",
        context_slice=ResearchContextSlice(selected_papers=["p1"] if selected_papers is None else selected_papers),
        dependency_summaries=[{"task_id": "compare-1", "summary": "p1 uses retrieval"}],
    )


def _context():
    papers = _papers()
    report = ResearchReport(
        report_id="old", task_id="task", topic="agents", generated_at="now",
        markdown="OLD FULL REPORT SHOULD STAY OUT OF THE PROMPT", highlights=["Unrelated p2 conclusion"],
    )
    return ResearchAgentToolContext(
        request=ResearchAgentRunRequest(message="Explain p1"), research_service=object(), graph_runtime=object(),
        task_response=ResearchTaskResponse(
            task=ResearchTask(task_id="task", topic="agents", created_at="now", updated_at="now"),
            papers=papers, report=report,
        ),
        worker_input=_task_context(),
        supervisor_instruction="Use the prior comparison to explain p1",
    )


def test_explicit_task_context_is_preserved_without_stale_decision_skill() -> None:
    context = _context()
    decision = SimpleNamespace(metadata={"skill_context": "stale decision skill", "active_skill_names": ["stale"]})
    expert = build_expert_context(context=context, decision=decision)

    assert expert["task_context"]["dependency_summaries"][0]["summary"] == "p1 uses retrieval"
    assert "skill_context" not in expert
    assert "active_skill_names" not in expert


@pytest.mark.asyncio
async def test_paper_analysis_llm_receives_task_and_prior_summary_without_unrelated_highlights() -> None:
    context = _context()
    analysis_input = build_paper_analysis_input(
        context=context, task_response=context.task_response, payload={"paper_ids": ["p1"]},
        papers=context.task_response.papers[:1],
    )
    llm = CapturingLLM({"answer": "p1 uses retrieval", "focus": "analysis"})
    agent = PaperAnalysisWorkflow(paper_analysis_tool=PaperAnalysisTool(llm_adapter=llm))

    await agent.analyze(
        question=analysis_input.resolved_question(), papers=analysis_input.papers,
        task_topic=analysis_input.task_topic, report_highlights=analysis_input.report_highlights,
        expert_context=analysis_input.metadata["expert_context"],
    )

    model_input = llm.calls[0]
    assert model_input["task_context"]["dependency_summaries"][0]["summary"] == "p1 uses retrieval"
    assert "p2" not in json.dumps(model_input)
    assert "OLD FULL REPORT" not in json.dumps(model_input)
    assert "research_service" not in json.dumps(model_input)


@pytest.mark.asyncio
async def test_writer_llm_receives_task_and_prior_summary_with_selected_paper_only() -> None:
    context = _context()
    review_input = build_review_draft_input(context=context)
    llm = CapturingLLM({"markdown": "## Background\nUse p1 [P1]"})
    writer = ResearchWriterAgent(paper_search_service=SimpleNamespace(survey_writer=SurveyWritingTool(llm_adapter=llm)))

    await writer.synthesize_async(review_input)

    model_input = llm.calls[0]
    assert model_input["task_context"]["dependency_summaries"][0]["summary"] == "p1 uses retrieval"
    assert "p2" not in json.dumps(model_input)
    assert "OLD FULL REPORT" not in json.dumps(model_input)
    assert "research_service" not in json.dumps(model_input)


@pytest.mark.asyncio
async def test_writer_explicit_task_redrafts_and_keeps_full_workspace_papers() -> None:
    context = _context()
    llm = CapturingLLM({"markdown": "## Background\nUse p1 [P1]"})
    writer = ResearchWriterAgent(paper_search_service=SimpleNamespace(survey_writer=SurveyWritingTool(llm_adapter=llm)))
    writer._quality_metrics = lambda report: {
        "passed": True, "word_count": 300, "has_citations": True, "has_key_sections": True, "issues": [],
    }

    result = await writer.run_action(context, SimpleNamespace(metadata={}))

    assert llm.calls
    assert [paper.paper_id for paper in result.state_delta.rebuild_execution_context_params["papers"]] == ["p1", "p2"]


@pytest.mark.asyncio
async def test_writer_unknown_document_scope_does_not_widen_to_workspace() -> None:
    context = _context()
    context.worker_input = _task_context([])
    review_input = build_review_draft_input(context=context)
    llm = CapturingLLM({"markdown": "Unexpected report"})
    writer = ResearchWriterAgent(paper_search_service=SimpleNamespace(survey_writer=SurveyWritingTool(llm_adapter=llm)))

    result = await writer.run_action(context, SimpleNamespace(metadata={}))

    assert review_input.curated_papers == []
    assert result.status == "skipped"
    assert result.metadata["reason"] == "no_papers"
    assert llm.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_source", ["message", "action_input"])
@pytest.mark.parametrize("document_id", ["d1", "unknown"])
async def test_analysis_run_action_obeys_prepared_document_scope(scope_source, document_id):
    from agents.research_supervisor_agent import ResearchSupervisorDecision
    from domain.schemas.agent_message import AgentMessage
    from runtime.research.worker_invocation import prepare_worker_invocation

    parent = _context()
    parent.worker_input = None
    for index, paper in enumerate(parent.papers, start=1):
        paper.metadata["document_id"] = f"d{index}"
    retrieval_calls = []

    from domain.schemas.evidence import EvidenceBundle
    from domain.schemas.retrieval import HybridRetrievalResult, RetrievalQuery
    from tooling.schemas import GraphSummaryToolOutput, HybridRetrieveToolOutput

    class RetrievalTools:
        async def retrieve(self, **kwargs):
            retrieval_calls.append(kwargs)
            return HybridRetrieveToolOutput(
                question=kwargs["question"], document_ids=kwargs["document_ids"],
                evidence_bundle=EvidenceBundle(),
                retrieval_result=HybridRetrievalResult(query=RetrievalQuery(
                    query=kwargs["question"], document_ids=kwargs["document_ids"],
                )),
            )

    async def query_graph_summary(**kwargs):
        retrieval_calls.append(kwargs)
        return GraphSummaryToolOutput()

    parent.graph_runtime = SimpleNamespace(
        retrieval_tools=RetrievalTools(), query_graph_summary=query_graph_summary,
    )
    scope = {"document_ids": [document_id]}
    decision = ResearchSupervisorDecision(
        action_name="analyze_papers", thought="", rationale="",
        action_input=scope if scope_source == "action_input" else {},
        metadata={"active_message": AgentMessage(
            task_id="analyze", agent_from="manager", agent_to="ResearchAnalystAgent",
            task_type="analyze_papers", instruction="Explain selected paper",
            payload=scope,
        )} if scope_source == "message" else {},
    )
    worker, decision = prepare_worker_invocation(context=parent, decision=decision, agent_results=[])
    llm = CapturingLLM({"answer": "Selected analysis", "focus": "analysis"})
    agent = PaperAnalysisWorkflow(paper_analysis_tool=PaperAnalysisTool(llm_adapter=llm))

    result = await agent.execute(worker, decision)

    if document_id == "unknown":
        assert result.status == "skipped"
        assert result.metadata["reason"] == "no_papers"
        assert llm.calls == []
        assert retrieval_calls == []
    else:
        assert result.status == "succeeded"
        assert len(retrieval_calls) == 2
        assert all(call["document_ids"] == ["d1"] for call in retrieval_calls)
        assert llm.calls
        assert '"p2"' not in json.dumps(llm.calls)
        assert [paper.paper_id for paper in result.state_delta.task_response.papers] == ["p1", "p2"]


@pytest.mark.parametrize("scope_source", ["active", "action_input", "message"])
def test_prepared_qa_scope_reaches_collection_selector(scope_source):
    from agents.research_supervisor_agent import ResearchSupervisorDecision
    from domain.schemas.agent_message import AgentMessage
    from domain.schemas.research_context import ResearchContext, ResearchExecutionContext
    from runtime.research.worker_invocation import prepare_worker_invocation
    from runtime.research.unified_action_adapters import build_collection_qa_input, resolve_active_message
    from tools.research.paper_selector import PaperSelectorService

    parent = _context()
    parent.worker_input = None
    parent.task.imported_document_ids = ["d1", "d2"]
    for index, paper in enumerate(parent.papers, start=1):
        paper.metadata["document_id"] = f"d{index}"
    parent.execution_context = ResearchExecutionContext(research_context=ResearchContext(active_papers=["p1"]))
    scope = {"document_ids": ["d1"]}
    decision = ResearchSupervisorDecision(
        action_name="answer_question", thought="", rationale="",
        action_input=scope if scope_source == "action_input" else {},
        metadata={"active_message": AgentMessage(
            task_id="qa", agent_from="manager", agent_to="ResearchAnalystAgent",
            task_type="answer_question", instruction="Explain selected paper", payload=scope,
        )} if scope_source == "message" else {},
    )
    worker, prepared = prepare_worker_invocation(context=parent, decision=decision, agent_results=[])
    qa_input = build_collection_qa_input(context=worker, task_id=parent.task.task_id,
                                       active_message=resolve_active_message(prepared))
    selected = PaperSelectorService().resolve_qa_scope(
        task=parent.task, papers=parent.papers,
        requested_paper_ids=qa_input.paper_ids, requested_document_ids=qa_input.document_ids,
    )
    assert selected.paper_ids == ["p1"]
    assert selected.document_ids == ["d1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["none", "document", "image"])
async def test_qa_empty_worker_paper_scope_skips_only_without_direct_route(route):
    from agents.research_analyst_agent import ResearchAnalystAgent

    context = _context()
    context.worker_input = _task_context([])
    if route == "document":
        context.request.selected_document_ids = ["standalone-doc"]
    if route == "image":
        context.request.chart_image_path = "/tmp/chart.png"
    agent = ResearchAnalystAgent()

    async def visual(**kwargs):
        return SimpleNamespace(search_new_figure=False)

    class ReachedDirectRoute(Exception):
        pass

    async def execute(*args, **kwargs):
        request = args[2]
        assert request.paper_ids == []
        assert request.document_ids == (["standalone-doc"] if route == "document" else [])
        raise ReachedDirectRoute

    agent._decide_visual_intent_for_context = visual
    agent.execute_qa = execute
    if route == "none":
        result = await agent.run_action(context, SimpleNamespace(metadata={}))
        assert result.status == "skipped"
        assert result.metadata["reason"] == "no_papers"
    else:
        with pytest.raises(ReachedDirectRoute):
            await agent.run_action(context, SimpleNamespace(metadata={}))


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["document", "image", "unknown_image"])
async def test_loaded_qa_direct_route_does_not_restore_unselected_workspace_papers(monkeypatch, route):
    from agents.research_analyst_agent import ResearchAnalystAgent
    from domain.schemas.research import ResearchTaskAskRequest
    from tools.research.paper_selector import PaperSelectorService
    from tools.research.visual_intent import VisualIntentDecision
    import tools.research.qa_tools as qa_tools

    context = _context()
    context.task.imported_document_ids = ["standalone-doc", "d2"]
    context.papers[1].metadata["document_id"] = "d2"
    request = ResearchTaskAskRequest(
        question="Explain this", document_ids=["standalone-doc"] if route == "document" else [],
        image_path="/tmp/chart.png" if route != "document" else None,
        paper_ids=["unknown"] if route == "unknown_image" else [],
        metadata={"routing_authority": "supervisor_llm", "preferred_qa_route": "chart_drilldown" if route != "document" else "document_drilldown",
                  "expert_context": {"task_context": _task_context([]).model_dump(mode="json")}},
    )

    async def visual(**kwargs):
        return VisualIntentDecision(intent="ambiguous", reuse_current_anchor=False, search_new_figure=False)

    async def infer(**kwargs):
        return None

    dependencies = SimpleNamespace(
        visual_anchor_tool=None, _load_cached_figure_payload=None, research_collection_qa_capability=None,
        paper_selector_service=PaperSelectorService(), visual_intent_router=SimpleNamespace(decide_async=visual),
        _extract_visual_anchor=lambda **kwargs: None, _restore_visual_anchor_from_workspace=lambda **kwargs: None,
        _infer_or_discover_visual_anchor=infer, research_context_manager=None, paper_reading_tool=None,
    )

    class ReachedScopedExecution(Exception):
        pass

    def capture(execution_context, **kwargs):
        assert kwargs["papers"] == []
        assert kwargs["document_ids"] == (["standalone-doc"] if route == "document" else [])
        raise ReachedScopedExecution

    monkeypatch.setattr(qa_tools, "scope_qa_execution_context", capture)
    with pytest.raises(ReachedScopedExecution):
        await ResearchAnalystAgent().execute_loaded_qa(
            dependencies, context.task, None, context.papers, request,
            graph_runtime=object(), execution_context=None,
        )


def test_qa_action_input_preserves_direct_document_without_matching_paper():
    from agents.research_supervisor_agent import ResearchSupervisorDecision
    from runtime.research.worker_invocation import prepare_worker_invocation
    from runtime.research.unified_action_adapters import build_collection_qa_input

    parent = _context()
    decision = ResearchSupervisorDecision(action_name="answer_question", thought="", rationale="",
                                          action_input={"document_ids": ["standalone-doc"]})
    worker, prepared = prepare_worker_invocation(context=parent, decision=decision, agent_results=[])
    qa_input = build_collection_qa_input(context=worker, task_id=parent.task.task_id,
                                        active_message=None, decision=prepared)
    assert qa_input.paper_ids == []
    assert qa_input.document_ids == ["standalone-doc"]


@pytest.mark.parametrize("metadata_source", ["paper", "research_context"])
def test_qa_paper_task_keeps_only_matching_request_documents(metadata_source):
    from domain.schemas.research_context import ResearchContext, ResearchContextPaperMeta, ResearchExecutionContext
    from runtime.research.unified_action_adapters import build_collection_qa_input

    context = _context()
    context.request.selected_document_ids = ["d1", "d2", "stale-doc"]
    if metadata_source == "paper":
        for index, paper in enumerate(context.papers, start=1):
            paper.metadata["document_id"] = f"d{index}"
    else:
        context.execution_context = ResearchExecutionContext(research_context=ResearchContext(
            imported_papers=[ResearchContextPaperMeta(paper_id=f"p{index}", title=f"p{index}", document_id=f"d{index}")
                             for index in (1, 2)],
        ))

    qa_input = build_collection_qa_input(
        context=context, task_id="task", active_message=None,
        decision=SimpleNamespace(action_input={"paper_ids": ["p1"]}),
    )

    assert qa_input.paper_ids == ["p1"]
    assert qa_input.document_ids == ["d1"]


@pytest.mark.parametrize("explicit_documents", [[], ["standalone-doc"]])
def test_qa_explicit_task_documents_override_request_selection(explicit_documents):
    from runtime.research.unified_action_adapters import build_collection_qa_input

    context = _context()
    context.request.selected_document_ids = ["d1"]
    context.papers[0].metadata["document_id"] = "d1"

    qa_input = build_collection_qa_input(
        context=context, task_id="task", active_message=None,
        decision=SimpleNamespace(action_input={"paper_ids": ["p1"], "document_ids": explicit_documents}),
    )

    assert qa_input.document_ids == explicit_documents
