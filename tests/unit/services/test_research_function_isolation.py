import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from domain.schemas.research_context import ResearchContext
from domain.schemas.research_functions import AnalyzePapersFunctionOutput
from tools.research.research_functions import ResearchFunctionService


def build_service(tmp_path, runtime=None):
    return ResearchFunctionService(
        research_service=SimpleNamespace(
            paper_search_service=SimpleNamespace(),
            report_service=SimpleNamespace(storage_root=tmp_path, papers_root=tmp_path),
        ),
        graph_runtime=runtime,
    )


def test_function_tools_do_not_import_or_construct_agents(tmp_path):
    root = Path(__file__).resolve().parents[3]
    tree = ast.parse((root / 'tools/research/research_functions.py').read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and name.startswith('agents') for name in imports)
    service = build_service(tmp_path)
    assert not hasattr(service, 'manager_agent')
    assert not hasattr(service, 'paper_analysis_agent')


def test_function_service_uses_runtime_llm_without_accessing_agents(tmp_path):
    class Runtime:
        llm_adapter = object()

        @property
        def plan_and_solve_reasoning_agent(self):
            raise AssertionError('tools must not access runtime agents')

    service = build_service(tmp_path, Runtime())
    assert service.paper_analysis_tool.llm_adapter is Runtime.llm_adapter


@pytest.mark.asyncio
async def test_analysis_and_recommendation_call_tool_directly(tmp_path):
    service = build_service(tmp_path)
    service.paper_analysis_tool.analyze_async = AsyncMock(return_value=AnalyzePapersFunctionOutput(answer='grounded'))
    # Poison former intermediary attributes; neither path may invoke them.
    service.paper_analysis_agent = object()
    output = await service.analyze_papers(question='Compare', paper_ids=[])
    assert output.answer == 'grounded'
    assert service.paper_analysis_tool.analyze_async.call_args.kwargs['evidence_hits'] == []
    await service.recommend_papers(based_on_context='topic', based_on_history=False)
    assert service.paper_analysis_tool.analyze_async.call_count == 2
    assert service.paper_analysis_tool.analyze_async.call_args.kwargs['question'] == 'topic'


@pytest.mark.asyncio
@pytest.mark.parametrize(('user_request', 'context', 'action'), [
    ('Research retrieval', ResearchContext(), 'search_literature'),
    ('Compare papers', ResearchContext(selected_papers=['a', 'b'], metadata={'task_id': 't'}), 'analyze_papers'),
    ('What is the method?', ResearchContext(selected_papers=['a'], metadata={'task_id': 't'}), 'answer_question'),
    ('Write a review', ResearchContext(known_conclusions=['evidence'], metadata={'task_id': 't'}), 'write_review'),
    ('Write a review', ResearchContext(), 'search_literature'),
])
async def test_decomposition_is_stateless_and_respects_available_context(tmp_path, user_request, context, action):
    service = build_service(tmp_path)
    service.manager_agent = object()
    before = context.model_dump()
    first = await service.decompose_task(user_request=user_request, context=context)
    second = await service.decompose_task(user_request=user_request, context=context)
    assert first == second
    assert first.task_plan[0].task_type == action
    assert first.metadata['decision_source'] == 'task_decomposition_tool'
    assert context.model_dump() == before


@pytest.mark.asyncio
async def test_decomposition_executes_through_registered_tool(tmp_path):
    from tooling.executor import ToolExecutor
    from tooling.registry import ToolRegistry
    from tooling.research_function_registry import ResearchFunctionRegistry

    service = build_service(tmp_path)
    registry = ToolRegistry()
    ResearchFunctionRegistry(registry).register('decompose_task', service.decompose_task)
    result = await ToolExecutor(registry).execute_tool_call('decompose_task', {'user_request': 'Research retrieval', 'context': {}})
    assert result.status == 'succeeded'
    assert result.output['task_plan'][0]['task_type'] == 'search_literature'
    assert result.output['metadata']['decision_source'] == 'task_decomposition_tool'


@pytest.mark.asyncio
@pytest.mark.parametrize('proposed_action', ['answer_question', 'sync_to_zotero', 'write_review'])
async def test_decomposition_validates_llm_proposal_against_available_actions(tmp_path, proposed_action):
    class LLM:
        async def generate_structured(self, *, prompt, input_data, response_model):
            return response_model.model_validate({'action_name': proposed_action, 'instruction': 'Inspect evidence'})

    service = build_service(tmp_path, SimpleNamespace(llm_adapter=LLM()))
    output = await service.decompose_task(user_request='Research retrieval', context=ResearchContext())
    assert output.task_plan[0].task_type == 'search_literature'
    assert output.metadata['decision_source'] == 'task_decomposition_tool'


@pytest.mark.asyncio
async def test_decomposition_uses_single_llm_proposal_without_dispatch(tmp_path):
    class LLM:
        calls = 0

        async def generate_structured(self, *, prompt, input_data, response_model):
            self.calls += 1
            return response_model.model_validate({'action_name': 'analyze_papers', 'instruction': 'Compare evidence for selected papers'})

    llm = LLM()
    service = build_service(tmp_path, SimpleNamespace(llm_adapter=llm))
    output = await service.decompose_task(user_request='Review findings', context=ResearchContext(selected_papers=['a', 'b'], metadata={'task_id': 't'}))
    assert output.task_plan[0].task_type == 'analyze_papers'
    assert output.task_plan[0].instruction == 'Compare evidence for selected papers'
    assert output.metadata['decision_source'] == 'task_decomposition_llm'
    assert llm.calls == 1
