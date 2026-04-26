import pytest

from tools.document_toolkit import DocumentAgent
from domain.schemas.document import DocumentPage, ParsedDocument, TextBlock


class PdfService:
    async def parse_document(self, file_path: str, document_id: str | None = None) -> ParsedDocument:
        return ParsedDocument(
            id=document_id or "doc1",
            filename=file_path,
            content_type="application/pdf",
            status="parsed",
            pages=[DocumentPage(id="p1", document_id=document_id or "doc1", page_number=1)],
        )


class OcrService:
    async def extract_text_blocks(self, page: DocumentPage) -> list[TextBlock]:
        return [TextBlock(id="tb1", document_id=page.document_id, page_id=page.id, page_number=1, text="hello")]


class LayoutService:
    async def locate_chart_candidates(self, page: DocumentPage) -> list[dict]:
        return [{"bbox": {"x0": 0, "y0": 0, "x1": 10, "y1": 10, "unit": "pixel"}}]


@pytest.mark.asyncio
async def test_document_agent_parse_summarize_and_locate() -> None:
    agent = DocumentAgent(PdfService(), OcrService(), LayoutService())
    doc = await agent.parse_document("a.pdf", "doc1")
    blocks = await agent.extract_text_blocks(doc.pages[0])
    page = doc.pages[0].model_copy(update={"text_blocks": blocks})
    summary = await agent.summarize_page(page)
    candidates = await agent.locate_chart_candidates(page)

    assert summary.summary == "hello"
    assert candidates[0].document_id == "doc1"
