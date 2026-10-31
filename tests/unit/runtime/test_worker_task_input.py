import json
from types import SimpleNamespace

from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research_context import (
    CompressedPaperSummary, QAPair, ResearchContext, ResearchContextPaperMeta,
    ResearchExecutionContext,
)


def build(action='analyze_papers', payload=None, dependencies=None, results=None):
    from runtime.research.worker_task_input import build_worker_task_input
    research = ResearchContext(
        research_topic='topic', selected_papers=['p2'], known_conclusions=['global secret'],
        imported_papers=[ResearchContextPaperMeta(paper_id=f'p{i}', title=f'Title {i}',
            document_id=f'd{i}', summary='x' * 10000, metadata={'secret': 'raw'}) for i in range(20)],
        paper_summaries=[CompressedPaperSummary(paper_id=f'p{i}', level=level, summary='s' * 10000,
            metadata={'secret': 'raw'}) for i in range(20) for level in ('document', 'section')],
        session_history=[QAPair(question='q', answer='a', citations=['p1']),
                         QAPair(question='unscoped', answer='secret'),
                         QAPair(question='mixed', answer='secret', citations=['p1', 'p2'])],
    )
    context = SimpleNamespace(execution_context=ResearchExecutionContext(research_context=research),
        request=SimpleNamespace(message='request', selected_paper_ids=[], selected_document_ids=[]))
    message = AgentMessage(task_id='current', agent_from='manager', agent_to='worker', task_type=action,
        instruction='do it', payload=payload or {}, depends_on=dependencies or [])
    decision = SimpleNamespace(action_name=action, action_input={}, metadata={'active_message': message})
    dto, artifacts = build_worker_task_input(context=context, decision=decision, agent_results=results or [])
    return dto, artifacts, research


def test_scopes_to_papers_and_only_attributable_history():
    dto, _, research = build(payload={'paper_ids': ['p1']})
    assert dto.context_slice.selected_papers == ['p1']
    assert [p.paper_id for p in dto.context_slice.imported_papers] == ['p1']
    assert [q.question for q in dto.context_slice.session_history] == ['q']
    assert dto.context_slice.known_conclusions == []
    assert len(dto.context_slice.imported_papers[0].summary) <= 2000
    assert 'secret' not in dto.model_dump_json()
    dto.context_slice.imported_papers[0].title = 'changed'
    assert research.imported_papers[1].title == 'Title 1'
    json.loads(dto.model_dump_json())


def test_explicit_ids_not_truncated_and_unknown_scope_is_empty():
    ids = [f'p{i}' for i in range(20)] + ['unknown']
    dto, _, _ = build(payload={'paper_ids': ids})
    assert dto.context_slice.selected_papers == ids
    dto, _, _ = build(payload={'paper_ids': ['unknown']})
    assert dto.context_slice.imported_papers == []
    dto, _, _ = build(payload={'paper_ids': []})
    assert len(dto.context_slice.selected_papers) == 20


def test_document_scope_and_action_specific_summary_level():
    dto, _, _ = build(payload={'document_ids': ['d1']})
    assert dto.context_slice.selected_papers == ['p1']
    assert dto.context_slice.summary_level == 'document'
    dto, _, _ = build(action='write_review', payload={'paper_ids': ['p1']})
    assert dto.context_slice.summary_level == 'section'
    assert all(s.level == 'section' for s in dto.context_slice.relevant_summaries)


def test_dependency_summaries_are_bounded_and_artifacts_detached():
    result = AgentResultMessage(task_id='dep', agent_from='a', agent_to='b', task_type='analyze_papers',
        status='succeeded', payload={'summary': 's' * 10000, 'report': {'secret': [1]}})
    unrelated = result.model_copy(update={'task_id': 'unrelated'})
    dto, artifacts, _ = build(dependencies=['dep', 'missing'], results=[result, unrelated])
    assert [s['task_id'] for s in dto.dependency_summaries] == ['dep', 'missing']
    assert dto.dependency_summaries[1]['status'] == 'missing'
    assert len(json.dumps(dto.dependency_summaries)) < 2500
    assert 'report' not in json.dumps(dto.dependency_summaries)
    assert len(dto.artifact_refs) == 1
    ref = dto.artifact_refs[0]
    assert ref.task_id == 'dep' and ref.kind == 'result'
    artifacts[ref.key]['report']['secret'].append(2)
    assert result.payload['report']['secret'] == [1]


def test_general_answer_can_ignore_research_context():
    dto, _, _ = build(action='general_answer', payload={'ignore_research_context': True})
    assert dto.context_slice.research_topic == ''
    assert dto.context_slice.selected_papers == []
    assert dto.context_slice.session_history == []


def test_candidates_before_import_and_full_stored_artifact():
    from domain.schemas.research import PaperCandidate
    from runtime.research.worker_task_input import build_worker_task_input
    context = SimpleNamespace(
        execution_context=ResearchExecutionContext(research_context=ResearchContext()),
        request=SimpleNamespace(message='request', selected_paper_ids=['new'], selected_document_ids=[]),
        papers=[PaperCandidate(paper_id='new', title='New candidate', source='arxiv', summary='abstract')],
        result_artifacts={'dep': {'full_answer': 'full content', 'nested': [1]}},
    )
    message = AgentMessage(task_id='current', agent_from='manager', agent_to='worker',
        task_type='analyze_papers', depends_on=['dep'])
    decision = SimpleNamespace(action_name='analyze_papers', action_input={}, metadata={'active_message': message})
    dependency = AgentResultMessage(task_id='dep', agent_from='worker', agent_to='manager',
        task_type='analyze_papers', status='succeeded', payload={'summary': 'brief'})
    dto, artifacts = build_worker_task_input(context=context, decision=decision, agent_results=[dependency])
    assert dto.context_slice.imported_papers[0].title == 'New candidate'
    assert artifacts[dto.artifact_refs[0].key]['full_answer'] == 'full content'
    artifacts[dto.artifact_refs[0].key]['nested'].append(2)
    assert context.result_artifacts['dep']['nested'] == [1]


def test_descriptive_paper_material_is_capped_but_ids_survive():
    dto, _, _ = build(payload={'paper_ids': [f'p{i}' for i in range(20)]})
    assert len(dto.context_slice.selected_papers) == 20
    assert len(dto.context_slice.imported_papers) <= 8
    assert len(dto.context_slice.relevant_summaries) <= 8


def test_discovery_omits_implicit_paper_content_but_respects_explicit_scope():
    dto, _, _ = build(action='search_literature')
    assert dto.context_slice.imported_papers == []
    assert dto.context_slice.session_history == []
    dto, _, _ = build(action='search_literature', payload={'paper_ids': ['p1']})
    assert dto.context_slice.imported_papers[0].paper_id == 'p1'


def test_writer_falls_back_to_document_summary():
    from runtime.research.worker_task_input import build_worker_task_input
    context = SimpleNamespace(
        execution_context=ResearchExecutionContext(research_context=ResearchContext(
            selected_papers=['p1'], paper_summaries=[CompressedPaperSummary(paper_id='p1', summary='document summary')],
        )), request=SimpleNamespace(message='write', selected_paper_ids=[], selected_document_ids=[]),
    )
    decision = SimpleNamespace(action_name='write_review', action_input={}, metadata={})
    dto, _ = build_worker_task_input(context=context, decision=decision, agent_results=[])
    assert dto.context_slice.relevant_summaries[0].summary == 'document summary'


def test_writer_without_selection_uses_available_workspace():
    from domain.schemas.research import PaperCandidate
    from runtime.research.worker_task_input import build_worker_task_input
    context = SimpleNamespace(
        execution_context=ResearchExecutionContext(research_context=ResearchContext()),
        request=SimpleNamespace(message='write', selected_paper_ids=[], selected_document_ids=[]),
        papers=[PaperCandidate(paper_id='new', title='New', source='arxiv')],
    )
    decision = SimpleNamespace(action_name='write_review', action_input={}, metadata={})
    dto, _ = build_worker_task_input(context=context, decision=decision, agent_results=[])
    assert dto.context_slice.selected_papers == ['new']
    assert dto.context_slice.imported_papers[0].title == 'New'


def test_workspace_without_execution_context_still_supplies_candidates():
    from domain.schemas.research import PaperCandidate
    from runtime.research.worker_task_input import build_worker_task_input
    context = SimpleNamespace(
        execution_context=None, task=SimpleNamespace(topic='New topic'),
        request=SimpleNamespace(message='write', selected_paper_ids=[], selected_document_ids=[]),
        papers=[PaperCandidate(paper_id='new', title='New', source='arxiv')],
    )
    decision = SimpleNamespace(action_name='write_review', action_input={}, metadata={})
    dto, _ = build_worker_task_input(context=context, decision=decision, agent_results=[])
    assert dto.context_slice.selected_papers == ['new']
    assert dto.context_slice.research_topic == 'New topic'
    assert context.execution_context is None
