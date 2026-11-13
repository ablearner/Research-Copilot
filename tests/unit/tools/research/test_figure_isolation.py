import ast
from pathlib import Path

import pytest

from domain.schemas.research import PaperCandidate, ResearchPaperFigurePreview


def test_qa_and_figure_services_do_not_depend_on_peer_chart_agent():
    root = Path(__file__).resolve().parents[4]
    for relative in (
        'agents/research_analyst_agent.py', 'tools/research/qa_tools.py',
        'tools/research/paper_figures.py',
        'services/research/qa_router.py', 'services/research/paper_operations.py',
    ):
        tree = ast.parse((root / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {'agents.chart_analysis_agent', 'agents.rag_react_qa_worker'}, relative
            if isinstance(node, ast.Attribute):
                assert node.attr != 'chart_analysis_agent', relative


@pytest.mark.asyncio
async def test_shared_figure_tool_reads_persisted_cache_without_parsing():
    from tools.research.paper_figures import ResearchPaperFigureTool

    paper = PaperCandidate(paper_id='p1', title='Paper', source='arxiv', metadata={'document_id': 'd1'})
    figure = ResearchPaperFigurePreview(figure_id='f1', paper_id='p1', document_id='d1', page_id='page1', page_number=1, chart_id='c1', image_path='/tmp/chart.png')
    def unexpected(**kwargs):
        pytest.fail('cache hit must not parse or persist again')
    result = await ResearchPaperFigureTool().list_paper_figures(
        task_id='t1', paper=paper, graph_runtime=None,
        load_cached_figure_payload=lambda **kwargs: {'document_id': 'd1', 'figures': [figure.model_dump()], 'warnings': ['cached']},
        persist_paper_figure_cache=unexpected, parse_imported_paper_document=unexpected,
    )
    assert result.figures == [figure]
    assert result.warnings == ['cached']
