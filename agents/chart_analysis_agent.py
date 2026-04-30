from __future__ import annotations

from pathlib import Path
from typing import Any

from domain.schemas.research import (
    AnalyzeResearchPaperFigureRequest,
    AnalyzeResearchPaperFigureResponse,
    PaperCandidate,
    ResearchPaperFigureListResponse,
    ResearchPaperFigurePreview,
)
from services.research.capabilities.paper_chart_analysis import PaperChartAnalyzer
from services.research.capabilities.visual_anchor import VisualAnchor
from services.research.research_knowledge_access import ResearchKnowledgeAccess
from tools.paper_figure_toolkit import PaperFigureAnalyzeTarget, PaperFigureTools


class ChartAnalysisAgent:
    """Top-level worker agent for chart selection and chart understanding."""

    name = "ChartAnalysisAgent"

    def __init__(self, *, llm_adapter: Any | None = None, storage_root: str | Path | None = None) -> None:
        self.llm_adapter = llm_adapter
        self.visual_anchor_skill = VisualAnchor(llm_adapter=llm_adapter)
        self.paper_chart_analysis_skill = PaperChartAnalyzer(llm_adapter=llm_adapter)
        self.paper_figure_tools = (
            PaperFigureTools(storage_root=storage_root) if storage_root is not None else None
        )

    async def infer_cached_visual_anchor(
        self,
        *,
        papers: list[PaperCandidate],
        document_ids: list[str],
        question: str,
        load_cached_figure_payload,
    ) -> dict[str, Any] | None:
        return await self.visual_anchor_skill.infer_cached_visual_anchor(
            papers=papers,
            document_ids=document_ids,
            question=question,
            load_cached_figure_payload=load_cached_figure_payload,
        )

    def resolve_visual_anchor_figure(
        self,
        *,
        papers: list[PaperCandidate],
        qa_metadata: dict[str, Any],
        load_cached_figure_payload,
    ) -> ResearchPaperFigurePreview | None:
        return self.visual_anchor_skill.resolve_visual_anchor_figure(
            papers=papers,
            qa_metadata=qa_metadata,
            load_cached_figure_payload=load_cached_figure_payload,
        )

    async def understand_chart(
        self,
        *,
        graph_runtime: Any,
        image_path: str,
        document_id: str,
        page_id: str,
        page_number: int | None,
        chart_id: str,
        session_id: str | None,
        context: dict[str, Any],
        skill_name: str | None,
    ) -> Any:
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        return await knowledge_access.understand_chart(
            image_path=image_path,
            document_id=document_id,
            page_id=page_id,
            page_number=page_number,
            chart_id=chart_id,
            session_id=session_id,
            context=context,
            skill_name=skill_name,
        )

    async def list_paper_figures(
        self,
        *,
        task_id: str,
        paper: PaperCandidate,
        graph_runtime: Any,
        load_cached_figure_payload,
        persist_paper_figure_cache,
        parse_imported_paper_document,
    ) -> ResearchPaperFigureListResponse:
        cached_payload = load_cached_figure_payload(paper=paper)
        if cached_payload is not None:
            return ResearchPaperFigureListResponse(
                task_id=task_id,
                paper_id=paper.paper_id,
                document_id=str(cached_payload.get("document_id") or paper.metadata.get("document_id") or ""),
                figures=[
                    ResearchPaperFigurePreview.model_validate(item)
                    for item in cached_payload.get("figures") or []
                ],
                warnings=[str(item) for item in (cached_payload.get("warnings") or [])],
            )
        if self.paper_figure_tools is None:
            raise RuntimeError("paper figure tools are not configured")
        parsed_document = await parse_imported_paper_document(paper=paper, graph_runtime=graph_runtime)
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        chart_candidates_by_page: dict[str, list[Any]] = {}
        for page in parsed_document.pages:
            candidates = await knowledge_access.locate_chart_candidates(page)
            chart_candidates_by_page[page.id] = list(candidates or [])
        previews, targets, warnings = await self.paper_figure_tools.build_figure_previews(
            paper_id=paper.paper_id,
            document_id=parsed_document.id,
            parsed_document=parsed_document,
            chart_candidates_by_page=chart_candidates_by_page,
        )
        target_map = {
            target.figure_id: target.model_dump(mode="json")
            for target in targets
        }
        enriched_previews = [
            preview.model_copy(
                update={
                    "metadata": {
                        **preview.metadata,
                        "analyze_target": target_map.get(preview.figure_id, {}),
                    }
                }
            )
            for preview in previews
        ]
        persist_paper_figure_cache(
            task_id=task_id,
            paper_id=paper.paper_id,
            document_id=parsed_document.id,
            storage_uri=str(paper.metadata.get("storage_uri") or "").strip(),
            figures=enriched_previews,
            targets=targets,
            warnings=warnings,
        )
        return ResearchPaperFigureListResponse(
            task_id=task_id,
            paper_id=paper.paper_id,
            document_id=parsed_document.id,
            figures=enriched_previews,
            warnings=warnings,
        )

    async def analyze_paper_figure(
        self,
        *,
        task_id: str,
        paper: PaperCandidate,
        request: AnalyzeResearchPaperFigureRequest,
        graph_runtime: Any,
        load_cached_figure_target,
        parse_imported_paper_document,
    ) -> AnalyzeResearchPaperFigureResponse:
        analyze_target = load_cached_figure_target(paper=paper, figure_id=request.figure_id)
        if analyze_target is None:
            parsed_document = await parse_imported_paper_document(paper=paper, graph_runtime=graph_runtime)
            analyze_target = self.resolve_figure_target(
                paper=paper,
                parsed_document=parsed_document,
                request=request,
            )
        chart_result = await self.understand_chart(
            graph_runtime=graph_runtime,
            image_path=analyze_target.image_path,
            document_id=analyze_target.document_id,
            page_id=analyze_target.page_id,
            page_number=analyze_target.page_number,
            chart_id=analyze_target.chart_id,
            session_id=None,
            context={
                "paper_id": paper.paper_id,
                "paper_title": paper.title,
                "figure_id": analyze_target.figure_id,
                "figure_source": analyze_target.source,
                "figure_title": analyze_target.metadata.get("title"),
                "figure_caption": analyze_target.metadata.get("caption"),
                "page_number": analyze_target.page_number,
                "bbox": analyze_target.bbox.model_dump(mode="json") if analyze_target.bbox else None,
                "selection_rationale": analyze_target.metadata.get("anchor_rationale"),
            },
            skill_name="paper_chart_analysis",
        )
        figure_context = {
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "figure_id": analyze_target.figure_id,
            "figure_source": analyze_target.source,
            "figure_title": analyze_target.metadata.get("title"),
            "figure_caption": analyze_target.metadata.get("caption"),
            "page_id": analyze_target.page_id,
            "page_number": analyze_target.page_number,
            "chart_id": analyze_target.chart_id,
            "image_path": analyze_target.image_path,
            "bbox": analyze_target.bbox.model_dump(mode="json") if analyze_target.bbox else None,
            "selection_rationale": analyze_target.metadata.get("anchor_rationale"),
        }
        answer, key_points = await self.paper_chart_analysis_skill.analyze_async(
            chart=chart_result.chart,
            question=request.question,
            figure_context=figure_context,
        )
        preview_data_url = None
        if self.paper_figure_tools is not None:
            preview_data_url = self.paper_figure_tools._image_to_data_url(analyze_target.image_path)
        figure_preview = ResearchPaperFigurePreview(
            figure_id=analyze_target.figure_id,
            paper_id=paper.paper_id,
            document_id=analyze_target.document_id,
            page_id=analyze_target.page_id,
            page_number=analyze_target.page_number,
            chart_id=analyze_target.chart_id,
            source=analyze_target.source if analyze_target.source in {"chart_candidate", "page_fallback"} else "chart_candidate",
            bbox=analyze_target.bbox,
            image_path=analyze_target.image_path,
            preview_data_url=preview_data_url,
            metadata=dict(analyze_target.metadata),
        )
        return AnalyzeResearchPaperFigureResponse(
            task_id=task_id,
            paper_id=paper.paper_id,
            figure=figure_preview,
            chart=chart_result.chart,
            graph_text=chart_result.graph_text,
            answer=answer,
            key_points=key_points,
            metadata={
                **chart_result.metadata,
                "figure_source": analyze_target.source,
                "figure_context": figure_context,
            },
        )

    def resolve_figure_target(
        self,
        *,
        paper: PaperCandidate,
        parsed_document: Any,
        request: AnalyzeResearchPaperFigureRequest,
    ) -> PaperFigureAnalyzeTarget:
        analyze_target = request.figure_id and str(request.figure_id).strip()
        if analyze_target:
            metadata_target = request.model_dump(mode="json")
        else:
            metadata_target = {}
        image_path = str(request.image_path or "").strip()
        page_id = str(request.page_id or "").strip()
        chart_id = str(request.chart_id or "").strip() or f"{page_id}_chart"
        page = next((item for item in parsed_document.pages if item.id == page_id), None)
        if page is None:
            raise KeyError(f"Page not found in parsed document: {page_id}")
        if not image_path:
            image_path = str(page.image_uri or "").strip()
        if not image_path:
            raise ValueError("Figure preview image is missing; please rediscover figures for this paper.")
        return PaperFigureAnalyzeTarget(
            figure_id=str(request.figure_id or f"{paper.paper_id}:{chart_id}"),
            paper_id=paper.paper_id,
            document_id=parsed_document.id,
            page_id=page.id,
            page_number=page.page_number,
            chart_id=chart_id,
            image_path=image_path,
            source="chart_candidate",
            metadata=metadata_target,
        )
