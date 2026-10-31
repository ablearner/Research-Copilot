from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agents.general_answer_agent import GeneralAnswerAgent
from agents.research_document_agent import ResearchDocumentAgent
from agents.chart_analysis_agent import ChartAnalysisAgent
from domain.schemas.research import ResearchAgentRunRequest, ResearchTask, ResearchTaskResponse
from domain.schemas.research_context import ResearchExecutionContext
from memory.research_context_manager import ResearchContextManager
from runtime.research.agent_protocol.base import ResearchAgentToolContext
from runtime.research.agent_protocol.mixins import persist_workspace_results
from runtime.research.supervisor_graph_runtime_core import ResearchSupervisorGraphRuntime


def context(**kwargs):
    return ResearchAgentToolContext(request=ResearchAgentRunRequest(message='hello'), research_service=SimpleNamespace(), graph_runtime=SimpleNamespace(), **kwargs)


@pytest.mark.asyncio
async def test_general_answer_only_changes_context_at_commit():
    ctx = context()
    result = await GeneralAnswerAgent().run_action(ctx, SimpleNamespace(metadata={}))
    assert ctx.general_answer is None
    assert result.state_delta.general_answer
    runtime = object.__new__(ResearchSupervisorGraphRuntime)
    runtime._apply_state_delta(context=ctx, delta=result.state_delta)
    assert ctx.general_answer == result.state_delta.general_answer


@pytest.mark.asyncio
@pytest.mark.parametrize('index_failure', [False, True])
async def test_document_failure_returns_deliberate_partial_progress(index_failure):
    parsed = SimpleNamespace(id='doc', pages=[])
    class Access:
        async def parse_document(self, **kwargs):
            if not index_failure:
                raise RuntimeError('parse failed')
            return parsed
        async def index_document(self, **kwargs):
            raise RuntimeError('index failed')
    ctx = context(knowledge_access=Access())
    ctx.request.document_file_path = '/tmp/input.pdf'
    result = await ResearchDocumentAgent().run_action(ctx, SimpleNamespace(metadata={}))
    assert result.status == 'failed'
    assert ctx.document_attempted is False
    assert ctx.parsed_document is None
    assert result.state_delta.document_attempted is True
    assert result.state_delta.parsed_document is (parsed if index_failure else None)
    runtime = object.__new__(ResearchSupervisorGraphRuntime)
    runtime._apply_state_delta(context=ctx, delta=result.state_delta)
    assert ctx.document_attempted is True
    assert ctx.parsed_document is (parsed if index_failure else None)


@pytest.mark.asyncio
async def test_missing_chart_defers_attempt_flag():
    ctx = context()
    result = await ChartAnalysisAgent().run_action(ctx, SimpleNamespace(metadata={}), task_type='supervisor_understand_chart')
    assert result.status == 'skipped'
    assert ctx.chart_attempted is False
    assert result.state_delta.chart_attempted is True


def test_workspace_calculation_does_not_mutate_execution_context():
    from domain.schemas.research_context import ResearchContext
    execution = ResearchExecutionContext(session_id='session', research_context=ResearchContext(session_id='session'))
    task = ResearchTask(task_id='task', topic='topic', created_at='now', updated_at='now')
    ctx = context(task_response=ResearchTaskResponse(task=task), execution_context=execution)
    ctx.research_service = SimpleNamespace(research_context_manager=ResearchContextManager(), build_context_slices=lambda *args, **kwargs: {}, save_task_state=Mock())
    before = asdict(execution)
    result = persist_workspace_results(ctx, compression_summary={'summary_count': 1}, persist=False)
    assert asdict(execution) == before
    assert ctx.task_response.task is task
    assert result.updated_execution_context is not execution
    ctx.research_service.save_task_state.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('persist,fail', [(False, False), (True, False), (True, True)])
async def test_qa_application_loads_inputs_and_commits_once(persist, fail):
    import inspect
    from services.research.qa_router import QAApplicationService
    from agents.research_qa_agent import QAExecutionResult
    from domain.schemas.research import ResearchTaskAskRequest
    task = SimpleNamespace(task_id='task', report_id='report')
    report, papers = object(), []
    store = Mock()
    store.load_task.return_value = task
    store.load_report.return_value = report
    store.load_papers.return_value = papers
    dependencies = {key: Mock() for key in inspect.signature(QAApplicationService).parameters}
    dependencies['report_service'] = store
    app = QAApplicationService(**dependencies)
    result = QAExecutionResult(response=object(), updated_task=task, updated_report=report, persist_params={'persist_research_update': {'session_id': 'session'}})
    class Executor:
        async def execute_loaded_qa(self, deps, loaded_task, loaded_report, loaded_papers, request, **kwargs):
            assert (loaded_task, loaded_report, loaded_papers) == (task, report, papers)
            assert kwargs["execution_context"] is app.build_execution_context.return_value
            if fail:
                raise RuntimeError('qa failed')
            return result
    if fail:
        with pytest.raises(RuntimeError, match='qa failed'):
            await app.execute_qa('task', ResearchTaskAskRequest(question='Question'), graph_runtime=object(), executor=Executor(), persist=persist)
    else:
        assert await app.execute_qa('task', ResearchTaskAskRequest(question='Question'), graph_runtime=object(), executor=Executor(), persist=persist) is result
    assert app.build_execution_context.call_count == 1
    expected = int(persist and not fail)
    assert app.save_task_state.call_count == expected
    assert store.save_report.call_count == expected
    assert app.memory_gateway.persist_research_update.call_count == expected


@pytest.mark.asyncio
async def test_import_preparation_defers_workflow_storage(tmp_path):
    from services.research.literature_research_service import LiteratureResearchService
    from adapters.storage.research_report_service import ResearchReportService
    from domain.schemas.research import ImportPapersRequest, PaperCandidate
    store = ResearchReportService(tmp_path)
    task = ResearchTask(task_id='task', topic='topic', created_at='now', updated_at='now')
    paper = PaperCandidate(paper_id='paper', title='Paper', source='arxiv', ingest_status='ingested', metadata={'document_id': 'doc', 'storage_uri': '/tmp/paper.pdf'})
    store.save_task(task)
    store.save_papers(task.task_id, [paper])
    svc = LiteratureResearchService(report_service=store, paper_search_service=SimpleNamespace(llm_adapter=None), paper_import_service=object())
    svc.save_task_state = Mock(wraps=svc.save_task_state)
    svc._update_research_memory = Mock()
    result = await svc.paper_service.prepare_paper_import(ImportPapersRequest(task_id='task', paper_ids=['paper']), graph_runtime=SimpleNamespace())
    assert result.response.skipped_count == 1
    assert result.task_response.task.imported_document_ids == ['doc']
    assert store.load_task('task').imported_document_ids == []
    svc.save_task_state.assert_not_called()
    svc._update_research_memory.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['succeeded', 'failed', 'skipped'])
async def test_runtime_commits_each_returned_delta_once(status):
    from unittest.mock import AsyncMock
    from agents.research_supervisor_agent import ResearchSupervisorDecision
    from runtime.research.agent_protocol.base import ResearchStateDelta, ResearchToolResult
    ctx = context()
    ctx.research_service = SimpleNamespace(append_runtime_event=Mock(), report_service=Mock(), save_task_state=Mock())
    task = ResearchTask(task_id='task', topic='topic', created_at='now', updated_at='now')
    delta = ResearchStateDelta(task=task, document_attempted=True)
    runtime = object.__new__(ResearchSupervisorGraphRuntime)
    runtime._execute_agent_run_action = AsyncMock(return_value=ResearchToolResult(status=status, observation='done', metadata={}, state_delta=delta))
    runtime._rebuild_execution_context = Mock()
    runtime._message_result_update = Mock(return_value={})
    decision = ResearchSupervisorDecision(action_name='understand_document', thought='read', rationale='read')
    await runtime._run_action_node({'context': ctx, 'current_decision': decision}, 'understand_document')
    assert ctx.document_attempted is True
    assert ctx.research_service.save_task_state.call_count == 1


@pytest.mark.asyncio
async def test_chart_failure_defers_attempt_until_commit():
    from unittest.mock import AsyncMock
    ctx = context()
    ctx.request.chart_image_path = '/tmp/chart.png'
    agent = ChartAnalysisAgent()
    agent.understand_chart = AsyncMock(side_effect=RuntimeError('chart failed'))
    result = await agent.run_action(ctx, SimpleNamespace(metadata={}), task_type='supervisor_understand_chart')
    assert result.status == 'failed'
    assert ctx.chart_attempted is False
    assert result.state_delta.chart_attempted is True


@pytest.mark.asyncio
async def test_compression_returns_new_execution_context_without_mutating_input():
    from agents.research_knowledge_agent import ResearchKnowledgeAgent
    from domain.schemas.research import PaperCandidate
    from domain.schemas.research_context import ResearchContext
    from tools.research import PaperReadingTool
    task = ResearchTask(task_id='task', topic='topic', created_at='now', updated_at='now')
    paper = PaperCandidate(paper_id='paper', title='Paper', abstract='Useful research evidence.', source='arxiv')
    execution = ResearchExecutionContext(session_id='session', research_context=ResearchContext(session_id='session'))
    ctx = context(task_response=ResearchTaskResponse(task=task, papers=[paper]), execution_context=execution)
    ctx.research_service = SimpleNamespace(research_context_manager=ResearchContextManager(), paper_reading_tool=PaperReadingTool(), build_context_slices=lambda *args, **kwargs: {}, memory_gateway=Mock())
    before = asdict(execution)
    result = await ResearchKnowledgeAgent().run_action(ctx, SimpleNamespace(metadata={}), task_type='compress_context')
    assert result.status == 'succeeded'
    assert asdict(execution) == before
    assert ctx.compressed_context_summary is None
    assert result.state_delta.execution_context.research_context.paper_summaries
    assert result.state_delta.memory_ops[0].op_type == 'save_context'
    runtime = object.__new__(ResearchSupervisorGraphRuntime)
    runtime._apply_state_delta(context=ctx, delta=result.state_delta)
    assert ctx.execution_context is result.state_delta.execution_context


@pytest.mark.asyncio
@pytest.mark.parametrize('sync_status', ['imported', 'failed'])
async def test_zotero_result_is_deferred_including_failed_sync(sync_status):
    from agents.research_knowledge_agent import ResearchKnowledgeAgent
    from domain.schemas.research import PaperCandidate
    from unittest.mock import AsyncMock
    task = ResearchTask(task_id='task', topic='topic', created_at='now', updated_at='now')
    paper = PaperCandidate(paper_id='paper', title='Paper', source='arxiv')
    ctx = context(task_response=ResearchTaskResponse(task=task, papers=[paper]))
    ctx.request.selected_paper_ids = ['paper']
    ctx.graph_runtime.research_function_service = SimpleNamespace(sync_paper_to_zotero=AsyncMock(return_value={'status': sync_status}))
    result = await ResearchKnowledgeAgent().run_action(ctx, SimpleNamespace(metadata={}), task_type='sync_to_zotero')
    assert result.status == ('succeeded' if sync_status == 'imported' else 'failed')
    assert ctx.zotero_sync_results is None
    assert result.state_delta.zotero_sync_results[0]['status'] == sync_status


@pytest.mark.asyncio
@pytest.mark.parametrize("with_conversation", [False, True])
async def test_specialist_import_workflow_commits_once(tmp_path, with_conversation):
    from agents.research_knowledge_agent import ResearchKnowledgeAgent
    from services.research.literature_research_service import LiteratureResearchService
    from adapters.storage.research_report_service import ResearchReportService
    from domain.schemas.research import PaperCandidate
    store = ResearchReportService(tmp_path)
    task = ResearchTask(task_id='task', topic='topic', created_at='now', updated_at='now')
    paper = PaperCandidate(paper_id='paper', title='Paper', source='arxiv', pdf_url='https://example.test/paper.pdf', ingest_status='selected')
    store.save_task(task)
    store.save_papers('task', [paper])
    svc = LiteratureResearchService(report_service=store, paper_search_service=SimpleNamespace(llm_adapter=None), paper_import_service=object())
    from domain.schemas.research import ImportedPaperResult
    async def import_one(candidate, **kwargs):
        return (
            ImportedPaperResult(paper_id='paper', title='Paper', status='imported', document_id='doc'),
            candidate.model_copy(update={'ingest_status': 'ingested', 'metadata': {'document_id': 'doc'}}),
        )
    svc.paper_service._import_single_paper = import_one
    svc.save_task_state = Mock(wraps=svc.save_task_state)
    svc._update_research_memory = Mock(wraps=svc._update_research_memory)
    from rag_runtime.memory import GraphSessionMemory
    ctx = context(task_response=ResearchTaskResponse(task=task, papers=[paper]))
    ctx.request.selected_paper_ids = ['paper']
    ctx.research_service = svc
    if with_conversation:
        from domain.schemas.research import CreateResearchConversationRequest
        conversation = svc.create_conversation(CreateResearchConversationRequest(topic='topic'))
        ctx.request.conversation_id = conversation.conversation.conversation_id
    ctx.graph_runtime.session_memory = GraphSessionMemory()
    result = await ResearchKnowledgeAgent().run_action(ctx, SimpleNamespace(metadata={}), task_type='import_papers')
    assert result.status == 'succeeded'
    assert ctx.import_attempted is False
    assert ctx.import_result is None
    assert store.load_task('task').imported_document_ids == []
    svc.save_task_state.assert_not_called()
    runtime = object.__new__(ResearchSupervisorGraphRuntime)
    runtime._apply_state_delta(context=ctx, delta=result.state_delta)
    runtime._persist_state_delta(context=ctx, delta=result.state_delta)
    assert ctx.import_attempted is True
    assert store.load_task('task').imported_document_ids == ['doc']
    assert svc.save_task_state.call_count == 1
    assert svc._update_research_memory.call_count == 1
    committed_session = ctx.graph_runtime.session_memory.load(ctx.request.conversation_id or 'task')
    assert committed_session.current_document_id == 'doc'
    assert ctx.execution_context.session_context['current_document_id'] == 'doc'
    assert ctx.execution_context.session_context['current_task_intent'] == 'research_import'
    assert ctx.execution_context.memory_hints['current_document_id'] == 'doc'


    if with_conversation:
        assert any(message['kind'] == 'import_result' for message in ctx.execution_context.conversation_context['recent_messages'])
        assert ctx.execution_context.conversation_context['selected_paper_ids'] == ['paper']


@pytest.mark.asyncio
@pytest.mark.parametrize("selection", [{"paper_ids": ["paper-1"]}, {"document_ids": ["doc-1"]}])
async def test_loaded_qa_never_reads_application_storage_and_preserves_scope(tmp_path, selection):
    from agents.research_qa_agent import ResearchQAAgent
    from services.research.literature_research_service import LiteratureResearchService
    from adapters.storage.research_report_service import ResearchReportService
    from domain.schemas.research import PaperCandidate, ResearchTaskAskRequest, ResearchReport, ResearchWorkspaceState
    from domain.schemas.api import QAResponse
    from domain.schemas.evidence import EvidenceBundle
    from unittest.mock import AsyncMock
    store = ResearchReportService(tmp_path)
    task = ResearchTask(task_id='task', topic='topic', created_at='now', updated_at='now', imported_document_ids=['doc-0', 'doc-1'], workspace=ResearchWorkspaceState(status_summary='Latest workspace', key_findings=['Latest finding']))
    report = ResearchReport(report_id='report', task_id='task', topic='topic', generated_at='now', markdown='Current report')
    papers = [PaperCandidate(paper_id=f'paper-{n}', title=f'Paper {n}', source='arxiv', ingest_status='ingested', metadata={'document_id': f'doc-{n}'}) for n in range(2)]
    svc = LiteratureResearchService(report_service=store, paper_search_service=SimpleNamespace(llm_adapter=None), paper_import_service=object())
    graph = SimpleNamespace()
    request = ResearchTaskAskRequest(question='Summarize the findings', **selection, metadata={'routing_authority': 'supervisor_llm', 'preferred_qa_route': 'collection_qa'})
    loaded = svc.build_execution_context(graph_runtime=graph, task=task, report=report, papers=papers)
    before = asdict(loaded)
    captured = []
    async def answer(**kwargs):
        captured.append(kwargs['execution_context'])
        return QAResponse(question=request.question, answer='Grounded answer', confidence=0.8, evidence_bundle=EvidenceBundle())
    svc.research_collection_qa_capability = SimpleNamespace(run_collection_qa=AsyncMock(side_effect=answer))
    def forbidden(*args, **kwargs):
        raise AssertionError('loaded executor accessed application storage')
    svc.qa_service.build_execution_context = forbidden
    svc.qa_service.report_service = SimpleNamespace(load_task=forbidden, load_report=forbidden, load_papers=forbidden, load_conversation=forbidden, load_messages=forbidden)
    svc.qa_service.memory_gateway = SimpleNamespace(load_user_profile=forbidden, hydrate_context=forbidden)
    # The executor receives context already hydrated by the application.
    result = await ResearchQAAgent().execute_loaded_qa(svc.qa_service, task, report, papers, request, graph_runtime=graph, execution_context=loaded)
    assert result.response.paper_ids == ['paper-1']
    scoped = captured[0]
    assert scoped.task_context['paper_titles'] == ['Paper 1']
    assert scoped.task_context['document_ids'] == ['doc-1']
    assert scoped.task_context['workspace_summary'] == 'Latest workspace'
    assert scoped.research_context.selected_papers == ['paper-1']
    assert 'Latest finding' in scoped.research_context.known_conclusions
    assert asdict(loaded) == before
