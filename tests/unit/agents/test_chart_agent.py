from types import SimpleNamespace

import pytest

from agents.chart_analysis_agent import ChartAnalysisAgent
from domain.schemas.research import PaperCandidate, ResearchPaperFigurePreview
from tools.chart_toolkit import ChartAgent


@pytest.mark.asyncio
async def test_chart_agent_parse_and_graph_text(mock_llm) -> None:
    agent = ChartAgent(mock_llm)
    chart = await agent.parse_chart("/tmp/chart.png", "doc1", "p1", 1, "chart1")

    assert chart.chart_type == "bar"
    assert "Revenue" in agent.explain_chart(chart)
    assert "chart_id: chart1" in agent.to_graph_text(chart)


@pytest.mark.asyncio
async def test_chart_analysis_selection_skips_excluded_anchor() -> None:
    agent = ChartAnalysisAgent()
    figures = [
        ResearchPaperFigurePreview(
            figure_id="paper-x:chart-1",
            paper_id="paper-x",
            document_id="doc-x",
            page_id="page-1",
            page_number=1,
            chart_id="chart-1",
            image_path="/tmp/missing-1.png",
        ),
        ResearchPaperFigurePreview(
            figure_id="paper-x:chart-2",
            paper_id="paper-x",
            document_id="doc-x",
            page_id="page-2",
            page_number=2,
            chart_id="chart-2",
            image_path="/tmp/missing-2.png",
        ),
    ]

    async def fake_anchor(**kwargs):
        return {"figure_id": "paper-x:chart-1"}

    agent.infer_cached_visual_anchor = fake_anchor
    context = SimpleNamespace(
        research_service=SimpleNamespace(_load_cached_figure_payload=lambda *, paper: None),
        graph_runtime=None,
    )
    paper = PaperCandidate(
        paper_id="paper-x",
        title="Paper X",
        source="arxiv",
        ingest_status="ingested",
        metadata={"document_id": "doc-x"},
    )

    selected = await agent._select_figure_via_anchor(
        question="这张不是，继续找",
        target_paper=paper,
        figures=figures,
        context=context,
        excluded_figure_ids={"paper-x:chart-1"},
    )

    assert selected.figure_id == "paper-x:chart-2"
