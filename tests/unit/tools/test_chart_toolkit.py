"""Tests for tools.chart_toolkit pure functions and data models."""

from tools.chart_toolkit import (
    ChartUnderstandInput,
    explain_chart,
    chart_to_graph_text,
)
from domain.schemas.chart import ChartSchema


class TestChartUnderstandInput:
    def test_minimal(self):
        inp = ChartUnderstandInput(
            image_path="/tmp/chart.png",
            document_id="doc1",
            page_id="p1",
            page_number=1,
            chart_id="c1",
        )
        assert inp.image_path == "/tmp/chart.png"

    def test_context_default(self):
        inp = ChartUnderstandInput(
            image_path="/tmp/chart.png",
            document_id="doc1",
            page_id="p1",
            page_number=1,
            chart_id="c1",
        )
        assert inp.context == {}


class TestExplainChart:
    def test_basic_chart(self):
        chart = ChartSchema(
            id="c1",
            document_id="doc1",
            page_id="p1",
            page_number=1,
            chart_type="line",
            summary="Rising trend",
        )
        result = explain_chart(chart)
        assert isinstance(result, str)
        assert len(result) > 0


class TestChartToGraphText:
    def test_basic_chart(self):
        chart = ChartSchema(
            id="c1",
            document_id="doc1",
            page_id="p1",
            page_number=1,
            chart_type="bar",
            summary="Comparison of methods",
        )
        result = chart_to_graph_text(chart)
        assert isinstance(result, str)
