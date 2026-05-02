"""Tests for tools.document_toolkit pure functions and data models."""

from tools.document_toolkit import (
    DocumentParseInput,
    PageSummary,
    page_text,
    trim_text,
)
from domain.schemas.document import DocumentPage, TextBlock


class TestDocumentParseInput:
    def test_minimal(self):
        inp = DocumentParseInput(file_path="/tmp/test.pdf")
        assert inp.file_path == "/tmp/test.pdf"
        assert inp.document_id is None

    def test_with_doc_id(self):
        inp = DocumentParseInput(file_path="/tmp/test.pdf", document_id="doc1")
        assert inp.document_id == "doc1"


class TestPageSummary:
    def test_create(self):
        ps = PageSummary(
            document_id="doc1",
            page_id="p1",
            page_number=1,
            summary="Overview of methods",
        )
        assert ps.summary == "Overview of methods"


class TestPageText:
    def test_basic_page(self):
        page = DocumentPage(
            id="p1",
            document_id="doc1",
            page_number=1,
            image_uri="/tmp/p1.png",
            text_blocks=[
                TextBlock(
                    id="tb1",
                    document_id="doc1",
                    page_id="p1",
                    page_number=1,
                    text="Block one.",
                    block_type="paragraph",
                ),
                TextBlock(
                    id="tb2",
                    document_id="doc1",
                    page_id="p1",
                    page_number=1,
                    text="Block two.",
                    block_type="paragraph",
                ),
            ],
        )
        result = page_text(page)
        assert "Block one." in result
        assert "Block two." in result

    def test_empty_blocks(self):
        page = DocumentPage(
            id="p1",
            document_id="doc1",
            page_number=1,
            text_blocks=[],
        )
        assert page_text(page) == ""


class TestTrimText:
    def test_short_text(self):
        assert trim_text("hello", 100) == "hello"

    def test_long_text(self):
        result = trim_text("a" * 200, 50)
        assert len(result) <= 50
        assert result.endswith("...")
