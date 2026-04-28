"""Tests for tools.graph_extraction_toolkit data models."""

from tools.graph_extraction_toolkit import (
    GraphFromTextBlocksInput,
    GraphFromChartInput,
    PageSummaryInput,
)


class TestPageSummaryInput:
    def test_create(self):
        inp = PageSummaryInput(page_id="p1", page_number=1, summary="Overview")
        assert inp.page_id == "p1"


class TestGraphFromTextBlocksInput:
    def test_minimal(self):
        inp = GraphFromTextBlocksInput(document_id="doc1")
        assert inp.document_id == "doc1"
        assert inp.text_blocks == []
        assert inp.page_summaries == []
