from pathlib import Path

import pytest

from adapters.local_runtime import LocalDocumentParser


def test_local_document_parser_splits_pdf_into_pages_without_binary_marker(tmp_path: Path) -> None:
    parser = LocalDocumentParser()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake payload that should never be surfaced as parsed text")

    parser._build_pdf_pages = lambda path, document_id: [  # type: ignore[method-assign]
        parser._build_fallback_page(path, document_id, reason="test"),
        parser._build_text_page(tmp_path / "page2.txt", document_id),
    ]
    (tmp_path / "page2.txt").write_text("Second page real text.", encoding="utf-8")

    parsed = __import__("asyncio").run(parser.parse_document(str(pdf_path), "doc_test"))

    assert len(parsed.pages) == 2
    assert parsed.pages[0].metadata["source_type"] == "fallback"
    assert parsed.pages[1].text_blocks[0].text == "Second page real text."
    assert "%PDF" not in parsed.pages[0].text_blocks[0].text


def test_local_document_parser_prefers_pymupdf_for_pdf_text(tmp_path: Path) -> None:
    parser = LocalDocumentParser()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake payload")

    py_page_text = tmp_path / "page_pymupdf.txt"
    py_page_text.write_text("Readable text from PyMuPDF.", encoding="utf-8")
    pdf_page_text = tmp_path / "page_pypdf.txt"
    pdf_page_text.write_text("Fallback text from pypdf.", encoding="utf-8")

    pymupdf_page = parser._build_text_page(py_page_text, "doc_test")
    pymupdf_page.metadata = {"source_type": "pdf", "text_engine": "pymupdf", "text_extracted": True}
    pypdf_page = parser._build_text_page(pdf_page_text, "doc_test")
    pypdf_page.metadata = {"source_type": "pdf", "text_engine": "pypdf", "text_extracted": True}

    parser._build_pdf_pages_with_pymupdf = lambda path, document_id: [pymupdf_page]  # type: ignore[method-assign]
    parser._build_pdf_pages_with_pypdf = lambda path, document_id: [pypdf_page]  # type: ignore[method-assign]

    parsed = __import__("asyncio").run(parser.parse_document(str(pdf_path), "doc_test"))

    assert parsed.pages[0].metadata["text_engine"] == "pymupdf"
    assert parsed.pages[0].text_blocks[0].text == "Readable text from PyMuPDF."


def test_local_document_parser_falls_back_to_pypdf_when_pymupdf_has_no_text(tmp_path: Path) -> None:
    parser = LocalDocumentParser()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake payload")

    empty_page = parser._build_fallback_page(pdf_path, "doc_test", reason="empty")
    empty_page.text_blocks = []
    empty_page.metadata = {"source_type": "pdf", "text_engine": "pymupdf", "text_extracted": False}

    pdf_page_text = tmp_path / "page_pypdf.txt"
    pdf_page_text.write_text("Recovered text from pypdf.", encoding="utf-8")
    pypdf_page = parser._build_text_page(pdf_page_text, "doc_test")
    pypdf_page.metadata = {"source_type": "pdf", "text_engine": "pypdf", "text_extracted": True}

    parser._build_pdf_pages_with_pymupdf = lambda path, document_id: [empty_page]  # type: ignore[method-assign]
    parser._build_pdf_pages_with_pypdf = lambda path, document_id: [pypdf_page]  # type: ignore[method-assign]

    parsed = __import__("asyncio").run(parser.parse_document(str(pdf_path), "doc_test"))

    assert parsed.pages[0].metadata["text_engine"] == "pypdf"
    assert parsed.pages[0].text_blocks[0].text == "Recovered text from pypdf."


def test_local_document_parser_renders_pdf_page_images_with_pymupdf(tmp_path: Path) -> None:
    fitz = pytest.importorskip("fitz")
    parser = LocalDocumentParser(storage_root=tmp_path / "storage")
    pdf_path = tmp_path / "renderable.pdf"

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Chart-like page text")
    document.save(str(pdf_path))
    document.close()

    parsed = __import__("asyncio").run(parser.parse_document(str(pdf_path), "doc_render"))

    assert len(parsed.pages) == 1
    assert parsed.pages[0].image_uri is not None
    assert Path(parsed.pages[0].image_uri).exists()
    assert parsed.pages[0].metadata["page_image_generated"] is True
