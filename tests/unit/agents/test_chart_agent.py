import pytest

from tools.chart_toolkit import ChartAgent


@pytest.mark.asyncio
async def test_chart_agent_parse_and_graph_text(mock_llm) -> None:
    agent = ChartAgent(mock_llm)
    chart = await agent.parse_chart("/tmp/chart.png", "doc1", "p1", 1, "chart1")

    assert chart.chart_type == "bar"
    assert "Revenue" in agent.explain_chart(chart)
    assert "chart_id: chart1" in agent.to_graph_text(chart)
