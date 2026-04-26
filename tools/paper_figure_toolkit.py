from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from domain.schemas.document import BoundingBox, DocumentPage, ParsedDocument
from domain.schemas.research import ResearchPaperFigurePreview
from tools.document_toolkit import ChartCandidate

logger = logging.getLogger(__name__)

_FIGURE_LINE_RE = re.compile(r"^\s*(figure|fig\.?|图)\s*[\-#:：]?\s*\d*", re.IGNORECASE)


class PaperFigureToolError(RuntimeError):
    pass


class PaperFigureAnalyzeTarget(BaseModel):
    figure_id: str
    paper_id: str
    document_id: str
    page_id: str
    page_number: int = Field(..., ge=1)
    chart_id: str
    image_path: str
    source: str = "chart_candidate"
    bbox: BoundingBox | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperFigureTools:
    def __init__(self, *, storage_root: str | Path) -> None:
        self.storage_root = Path(storage_root).expanduser().resolve()

    async def build_figure_previews(
        self,
        *,
        paper_id: str,
        document_id: str,
        parsed_document: ParsedDocument,
        chart_candidates_by_page: dict[str, list[ChartCandidate]],
        max_figures: int = 8,
    ) -> tuple[list[ResearchPaperFigurePreview], list[PaperFigureAnalyzeTarget], list[str]]:
        previews: list[ResearchPaperFigurePreview] = []
        targets: list[PaperFigureAnalyzeTarget] = []
        warnings: list[str] = []
        source_path = str(parsed_document.metadata.get("source_path") or "").strip()

        for page in parsed_document.pages:
            candidates = chart_candidates_by_page.get(page.id) or []
            if not candidates:
                fallback = self._page_fallback_candidate(page)
                if fallback is not None:
                    candidates = [fallback]
            for index, candidate in enumerate(candidates, start=1):
                if len(previews) >= max_figures:
                    return previews, targets, warnings
                resolved_chart_id = str(candidate.id or f"{page.id}_chart_{index}")
                image_path = self._prepare_image_path(
                    parsed_document=parsed_document,
                    page=page,
                    candidate=candidate,
                    chart_id=resolved_chart_id,
                    source_path=source_path,
                )
                if not image_path:
                    warnings.append(f"未能为 page={page.id} 的图表候选生成预览图。")
                    continue
                preview = ResearchPaperFigurePreview(
                    figure_id=f"{paper_id}:{resolved_chart_id}",
                    paper_id=paper_id,
                    document_id=document_id,
                    page_id=page.id,
                    page_number=page.page_number,
                    chart_id=resolved_chart_id,
                    title=self._title_from_candidate(candidate),
                    caption=self._caption_from_candidate(candidate),
                    source="chart_candidate" if candidate.bbox is not None else "page_fallback",
                    bbox=candidate.bbox,
                    image_path=image_path,
                    preview_data_url=self._image_to_data_url(image_path),
                    metadata=dict(candidate.metadata),
                )
                previews.append(preview)
                targets.append(
                    PaperFigureAnalyzeTarget(
                        figure_id=preview.figure_id,
                        paper_id=paper_id,
                        document_id=document_id,
                        page_id=page.id,
                        page_number=page.page_number,
                        chart_id=resolved_chart_id,
                        image_path=image_path,
                        source=preview.source,
                        bbox=candidate.bbox,
                        metadata={
                            **dict(candidate.metadata),
                            "title": preview.title,
                            "caption": preview.caption,
                            "preview_data_url_available": bool(preview.preview_data_url),
                        },
                    )
                )
        return previews, targets, warnings

    def _page_fallback_candidate(self, page: DocumentPage) -> ChartCandidate | None:
        if not page.image_uri:
            return None
        page_text = " ".join(block.text for block in page.text_blocks if block.text.strip()).lower()
        has_figure_marker = any(marker in page_text for marker in ("figure", "fig.", "图", "chart", "plot"))
        if not has_figure_marker and page.page_number > 4:
            return None
        fallback_metadata = {"fallback": "page_image"}
        inferred_title, inferred_caption = self._infer_page_fallback_figure_text(page)
        if inferred_title:
            fallback_metadata["title"] = inferred_title
        if inferred_caption:
            fallback_metadata["caption"] = inferred_caption
        return ChartCandidate(
            id=f"{page.id}_page_fallback",
            document_id=page.document_id,
            page_id=page.id,
            page_number=page.page_number,
            bbox=None,
            image_uri=page.image_uri,
            confidence=0.2,
            metadata=fallback_metadata,
        )

    def _infer_page_fallback_figure_text(self, page: DocumentPage) -> tuple[str | None, str | None]:
        candidate_lines: list[str] = []
        for block in page.text_blocks:
            text = str(block.text or "").strip()
            if not text:
                continue
            if block.block_type in {"caption", "title"}:
                candidate_lines.append(text)
                continue
            if _FIGURE_LINE_RE.match(text):
                candidate_lines.append(text)
        if not candidate_lines:
            return None, None

        caption = next((line for line in candidate_lines if _FIGURE_LINE_RE.match(line)), candidate_lines[0])
        title = self._title_from_caption_line(caption)
        return title, caption[:500]

    def _title_from_caption_line(self, caption: str) -> str | None:
        normalized = caption.strip()
        if not normalized:
            return None
        stripped = re.sub(r"^\s*(figure|fig\.?|图)\s*[\-#:：]?\s*\d*\s*", "", normalized, flags=re.IGNORECASE)
        stripped = stripped.lstrip(".:：- ").strip()
        return stripped[:200] if stripped else None

    def _prepare_image_path(
        self,
        *,
        parsed_document: ParsedDocument,
        page: DocumentPage,
        candidate: ChartCandidate,
        chart_id: str,
        source_path: str,
    ) -> str | None:
        if candidate.bbox is not None and source_path.lower().endswith(".pdf"):
            cropped = self._crop_pdf_region(
                source_path=source_path,
                page=page,
                bbox=candidate.bbox,
                document_id=parsed_document.id,
                chart_id=chart_id,
            )
            if cropped is not None:
                return cropped
        return str(candidate.image_uri or page.image_uri or "").strip() or None

    def _crop_pdf_region(
        self,
        *,
        source_path: str,
        page: DocumentPage,
        bbox: BoundingBox,
        document_id: str,
        chart_id: str,
    ) -> str | None:
        try:
            import fitz

            document = fitz.open(source_path)
            try:
                pdf_page = document.load_page(page.page_number - 1)
                page_rect = pdf_page.rect
                clip = self._bbox_to_pdf_rect(page=page, bbox=bbox, page_rect=page_rect)
                if clip is None or clip.width <= 1 or clip.height <= 1:
                    return None
                target_path = self.storage_root / "chart_crops" / document_id / f"{chart_id}.png"
                target_path.parent.mkdir(parents=True, exist_ok=True)
                pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
                pixmap.save(str(target_path))
                return str(target_path)
            finally:
                document.close()
        except Exception:
            logger.warning(
                "Failed to crop paper figure from PDF",
                extra={"document_id": document_id, "page_id": page.id, "chart_id": chart_id},
                exc_info=True,
            )
            return None

    def _bbox_to_pdf_rect(self, *, page: DocumentPage, bbox: BoundingBox, page_rect: Any) -> Any | None:
        try:
            import fitz

            if bbox.unit == "relative":
                return fitz.Rect(
                    bbox.x0 * page_rect.width,
                    bbox.y0 * page_rect.height,
                    bbox.x1 * page_rect.width,
                    bbox.y1 * page_rect.height,
                )
            if bbox.unit == "point":
                return fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
            if bbox.unit == "pixel":
                page_image_uri = str(page.image_uri or "").strip()
                if not page_image_uri:
                    return None
                pixmap = fitz.Pixmap(page_image_uri)
                return fitz.Rect(
                    bbox.x0 / max(pixmap.width, 1) * page_rect.width,
                    bbox.y0 / max(pixmap.height, 1) * page_rect.height,
                    bbox.x1 / max(pixmap.width, 1) * page_rect.width,
                    bbox.y1 / max(pixmap.height, 1) * page_rect.height,
                )
        except Exception:
            return None
        return None

    def _image_to_data_url(self, image_path: str) -> str | None:
        try:
            path = Path(image_path)
            if not path.exists():
                return None
            mime = "image/png"
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".webp":
                mime = "image/webp"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{encoded}"
        except Exception:
            logger.warning("Failed to encode paper figure preview", extra={"image_path": image_path}, exc_info=True)
            return None

    def _title_from_candidate(self, candidate: ChartCandidate) -> str | None:
        title = candidate.metadata.get("title")
        return str(title).strip() if title is not None and str(title).strip() else None

    def _caption_from_candidate(self, candidate: ChartCandidate) -> str | None:
        caption = candidate.metadata.get("caption")
        return str(caption).strip() if caption is not None and str(caption).strip() else None
