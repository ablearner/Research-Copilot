import logging
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter
from domain.schemas.document import BoundingBox, DocumentPage, ParsedDocument, TextBlock
from rag_runtime.services.layout_service import LayoutService
from rag_runtime.services.ocr_service import OcrService
from rag_runtime.services.pdf_service import PdfService

logger = logging.getLogger(__name__)


class DocumentAgentError(RuntimeError):
    """Raised when document understanding fails."""


class PageSummary(BaseModel):
    document_id: str
    page_id: str
    page_number: int = Field(..., ge=1)
    summary: str
    source: Literal["text_blocks", "metadata", "empty"] = "text_blocks"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChartCandidate(BaseModel):
    id: str
    document_id: str
    page_id: str
    page_number: int = Field(..., ge=1)
    bbox: BoundingBox | None = None
    image_uri: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentParseInput(BaseModel):
    file_path: str
    document_id: str | None = None


class PageSummarizeInput(BaseModel):
    page: DocumentPage
    max_chars: int = 800


class TextBlockExtractionInput(BaseModel):
    page: DocumentPage


class ChartCandidateInput(BaseModel):
    page: DocumentPage


async def parse_document_run(
    *,
    pdf_service: PdfService,
    file_path: str,
    document_id: str | None = None,
) -> ParsedDocument:
    try:
        parsed = await pdf_service.parse_document(file_path=file_path, document_id=document_id)
        logger.info("Parsed document", extra={"document_id": parsed.id, "page_count": len(parsed.pages)})
        return parsed
    except Exception as exc:
        logger.exception("Failed to parse document", extra={"file_path": file_path})
        raise DocumentAgentError("Failed to parse document") from exc


def page_text(page: DocumentPage) -> str:
    return "\n".join(block.text.strip() for block in page.text_blocks if block.text.strip())


def trim_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


async def summarize_page_run(
    *,
    page: DocumentPage,
    max_chars: int = 800,
) -> PageSummary:
    try:
        text = page_text(page)
        if not text:
            return PageSummary(
                document_id=page.document_id,
                page_id=page.id,
                page_number=page.page_number,
                summary="",
                source="empty",
            )
        return PageSummary(
            document_id=page.document_id,
            page_id=page.id,
            page_number=page.page_number,
            summary=trim_text(text, max_chars=max_chars),
            source="text_blocks",
            metadata={"text_block_count": len(page.text_blocks)},
        )
    except Exception as exc:
        logger.exception("Failed to summarize page", extra={"page_id": page.id})
        raise DocumentAgentError("Failed to summarize page") from exc


async def extract_text_blocks_run(
    *,
    ocr_service: OcrService,
    page: DocumentPage,
) -> list[TextBlock]:
    if page.text_blocks:
        return page.text_blocks
    try:
        blocks = await ocr_service.extract_text_blocks(page)
        logger.info("Extracted text blocks", extra={"page_id": page.id, "text_block_count": len(blocks)})
        return blocks
    except Exception as exc:
        logger.exception("Failed to extract text blocks", extra={"page_id": page.id})
        raise DocumentAgentError("Failed to extract text blocks") from exc


def normalize_chart_candidate(
    *,
    page: DocumentPage,
    raw_candidate: dict[str, Any],
    index: int,
) -> ChartCandidate:
    bbox = raw_candidate.get("bbox")
    return ChartCandidate(
        id=str(raw_candidate.get("id") or f"{page.id}_chart_candidate_{index + 1}"),
        document_id=page.document_id,
        page_id=page.id,
        page_number=page.page_number,
        bbox=BoundingBox.model_validate(bbox) if bbox else None,
        image_uri=raw_candidate.get("image_uri") or page.image_uri,
        confidence=raw_candidate.get("confidence"),
        metadata={key: value for key, value in raw_candidate.items() if key not in {"bbox", "image_uri"}},
    )


async def locate_chart_candidates_run(
    *,
    layout_service: LayoutService,
    page: DocumentPage,
) -> list[ChartCandidate]:
    try:
        raw_candidates = await layout_service.locate_chart_candidates(page)
        candidates = [
            normalize_chart_candidate(page=page, raw_candidate=raw_candidate, index=index)
            for index, raw_candidate in enumerate(raw_candidates)
        ]
        logger.info("Located chart candidates", extra={"page_id": page.id, "candidate_count": len(candidates)})
        return candidates
    except Exception as exc:
        logger.exception("Failed to locate chart candidates", extra={"page_id": page.id})
        raise DocumentAgentError("Failed to locate chart candidates") from exc


class DocumentAgent:
    def __init__(
        self,
        pdf_service: PdfService,
        ocr_service: OcrService,
        layout_service: LayoutService,
        llm_adapter: BaseLLMAdapter | None = None,
    ) -> None:
        self.pdf_service = pdf_service
        self.ocr_service = ocr_service
        self.layout_service = layout_service
        self.llm_adapter = llm_adapter
        self.parse_document_tool = StructuredTool.from_function(
            coroutine=self.parse_document,
            name="parse_document",
            description="Parse a document file into ParsedDocument.",
            args_schema=DocumentParseInput,
        )
        self.extract_text_blocks_tool = StructuredTool.from_function(
            coroutine=self.extract_text_blocks,
            name="extract_text_blocks",
            description="Extract text blocks for a document page.",
            args_schema=TextBlockExtractionInput,
        )
        self.locate_chart_candidates_tool = StructuredTool.from_function(
            coroutine=self.locate_chart_candidates,
            name="locate_chart_candidates",
            description="Locate chart candidates from a document page layout.",
            args_schema=ChartCandidateInput,
        )

    async def parse_document(self, file_path: str, document_id: str | None = None) -> ParsedDocument:
        return await parse_document_run(
            pdf_service=self.pdf_service,
            file_path=file_path,
            document_id=document_id,
        )

    async def summarize_page(
        self,
        page: DocumentPage,
        max_chars: int = 800,
    ) -> PageSummary:
        return await summarize_page_run(page=page, max_chars=max_chars)

    async def extract_text_blocks(self, page: DocumentPage) -> list[TextBlock]:
        return await extract_text_blocks_run(ocr_service=self.ocr_service, page=page)

    async def locate_chart_candidates(self, page: DocumentPage) -> list[ChartCandidate]:
        return await locate_chart_candidates_run(layout_service=self.layout_service, page=page)


DocumentTools = DocumentAgent
DocumentToolsError = DocumentAgentError
