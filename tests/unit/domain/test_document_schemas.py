from domain.schemas.document import BoundingBox, DocumentPage, ParsedDocument, TextBlock


def test_document_schema_json_schema() -> None:
    assert ParsedDocument.model_json_schema()["title"] == "ParsedDocument"


def test_document_models_validate() -> None:
    bbox = BoundingBox(x0=0, y0=0, x1=10, y1=20)
    block = TextBlock(
        id="tb1",
        document_id="doc1",
        page_id="p1",
        page_number=1,
        text="hello",
        bbox=bbox,
    )
    page = DocumentPage(id="p1", document_id="doc1", page_number=1, text_blocks=[block])
    doc = ParsedDocument(
        id="doc1",
        filename="a.pdf",
        content_type="application/pdf",
        status="parsed",
        pages=[page],
    )
    assert doc.pages[0].text_blocks[0].bbox == bbox
