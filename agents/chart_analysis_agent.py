from __future__ import annotations

import logging
from pathlib import Path
import re
import shutil
from typing import TYPE_CHECKING, Any

from domain.schemas.research import (
    AnalyzeResearchPaperFigureRequest,
    AnalyzeResearchPaperFigureResponse,
    PaperCandidate,
    ResearchPaperFigureListResponse,
    ResearchPaperFigurePreview,
)
from tools.research.paper_chart_analysis import PaperChartAnalyzer
from tools.research.visual_anchor import VisualAnchor
from tools.research.knowledge_access import ResearchKnowledgeAccess
from tools.paper_figure_toolkit import PaperFigureAnalyzeTarget, PaperFigureTools

if TYPE_CHECKING:
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )

logger = logging.getLogger(__name__)


class ChartAnalysisAgent:
    """Top-level worker agent for chart selection and chart understanding."""

    name = "ChartAnalysisAgent"

    def __init__(
        self,
        *,
        llm_adapter: Any | None = None,
        storage_root: str | Path | None = None,
    ) -> None:
        self.llm_adapter = llm_adapter
        self.visual_anchor_skill = VisualAnchor(llm_adapter=llm_adapter)
        self.paper_chart_analysis_skill = PaperChartAnalyzer(llm_adapter=llm_adapter)
        self.paper_figure_tools = (
            PaperFigureTools(storage_root=storage_root) if storage_root is not None else None
        )

    # ------------------------------------------------------------------
    # New unified entry point (SpecialistAgent protocol)
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
        *,
        task_type: str = "supervisor_understand_chart",
    ) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import ResearchToolResult

        if task_type == "supervisor_understand_chart":
            return await self._run_understand_chart(context=context, decision=decision)
        if task_type == "analyze_paper_figures":
            return await self._run_analyze_paper_figures(context=context, decision=decision)
        return ResearchToolResult(
            status="skipped",
            observation=f"ChartAnalysisAgent does not support task_type={task_type}",
            metadata={"reason": "unsupported_task_type"},
        )

    async def _run_understand_chart(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import ResearchToolResult
        from runtime.research.unified_action_adapters import (
            build_chart_understanding_input,
            build_chart_understanding_output,
        )

        context.chart_attempted = True
        chart_input = build_chart_understanding_input(context=context, decision=decision)
        if not chart_input.image_path:
            return ResearchToolResult(
                status="skipped",
                observation="no chart_image_path was provided for chart understanding",
                metadata={"reason": "missing_chart_image_path"},
            )
        chart_context = dict(chart_input.context or {})
        if context.supervisor_instruction:
            chart_context["supervisor_instruction"] = context.supervisor_instruction
        chart_result = await self.understand_chart(
            graph_runtime=context.graph_runtime,
            image_path=chart_input.image_path,
            document_id=chart_input.document_id,
            page_id=chart_input.page_id,
            page_number=chart_input.page_number,
            chart_id=chart_input.chart_id,
            session_id=chart_input.session_id,
            context=chart_context,
            skill_name=chart_input.skill_name,
        )
        context.chart_result = chart_result
        chart = getattr(chart_result, "chart", None)
        output = build_chart_understanding_output(
            chart_result=chart_result,
            chart_input=chart_input,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"chart understood; chart_id={getattr(chart, 'id', chart_input.chart_id)}; "
                f"chart_type={getattr(chart, 'chart_type', 'unknown')}"
            ),
            metadata=output.to_metadata(),
        )

    async def _run_analyze_paper_figures(self, *, context: ResearchAgentToolContext, decision: Any) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import ResearchToolResult
        from runtime.research.unified_action_adapters import resolve_active_message

        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for paper figure analysis",
                metadata={"reason": "missing_task"},
            )
        active_message = resolve_active_message(decision)
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        question = str(payload.get("question") or context.request.message or "").strip()

        paper_ids = [
            str(item).strip()
            for item in (payload.get("paper_ids") or context.request.selected_paper_ids)
            if str(item).strip()
        ]
        imported_papers = [
            p for p in task_response.papers
            if str(p.metadata.get("document_id") or "").strip()
            and str(p.metadata.get("storage_uri") or "").strip()
        ]
        target_papers = [p for p in imported_papers if p.paper_id in paper_ids] if paper_ids else imported_papers
        if not target_papers:
            logger.warning(
                "analyze_paper_figures: no target papers; imported_papers=%d; paper_ids=%s; all_papers=%s",
                len(imported_papers),
                paper_ids,
                [(p.paper_id, p.ingest_status, bool(p.metadata.get("document_id")), bool(p.metadata.get("storage_uri"))) for p in task_response.papers[:5]],
            )
            return ResearchToolResult(
                status="skipped",
                observation="no imported paper with a local document is available for figure analysis",
                metadata={"reason": "no_imported_papers"},
            )
        target_paper = target_papers[0]
        logger.info("analyze_paper_figures: target_paper=%s doc_id=%s", target_paper.paper_id, target_paper.metadata.get("document_id"))

        try:
            figure_list = await context.research_service.list_paper_figures(
                task_response.task.task_id,
                target_paper.paper_id,
                graph_runtime=context.graph_runtime,
            )
        except Exception as exc:
            logger.warning("Failed to list paper figures", exc_info=True)
            return ResearchToolResult(
                status="failed",
                observation=f"failed to extract figures from paper: {exc}",
                metadata={"reason": "list_figures_failed"},
            )

        if not figure_list.figures:
            return ResearchToolResult(
                status="skipped",
                observation=f"no figures found in paper '{target_paper.title}'",
                metadata={"reason": "no_figures", "paper_id": target_paper.paper_id},
            )

        best_figure = await self._select_figure_via_anchor(
            question=question or "",
            target_paper=target_paper,
            figures=figure_list.figures,
            context=context,
        )

        figure_request = AnalyzeResearchPaperFigureRequest(
            figure_id=best_figure.figure_id,
            page_id=best_figure.page_id,
            chart_id=best_figure.chart_id,
            image_path=best_figure.image_path,
            question=question or None,
        )
        try:
            analysis_response = await context.research_service.analyze_paper_figure(
                task_response.task.task_id,
                target_paper.paper_id,
                figure_request,
                graph_runtime=context.graph_runtime,
                supervisor_instruction=context.supervisor_instruction,
            )
        except Exception as exc:
            logger.warning("Failed to analyze paper figure", exc_info=True)
            return ResearchToolResult(
                status="failed",
                observation=f"figure analysis failed: {exc}",
                metadata={"reason": "analyze_figure_failed"},
            )

        exported_image_path = self._export_figure_image(
            analysis_response=analysis_response,
            task_id=task_response.task.task_id,
        )
        if exported_image_path and analysis_response.chart and hasattr(analysis_response.chart, "metadata"):
            analysis_response.chart.metadata["image_path"] = exported_image_path

        context.chart_result = analysis_response
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"paper figure analysis completed; paper='{target_paper.title}'; "
                f"figure={best_figure.figure_id}; answer_length={len(analysis_response.answer)}"
            ),
            metadata={
                "paper_id": target_paper.paper_id,
                "figure_id": best_figure.figure_id,
                "answer": analysis_response.answer,
                "key_points": analysis_response.key_points,
                "chart_type": getattr(analysis_response.chart, "chart_type", None),
                "image_path": exported_image_path,
            },
        )

    async def _select_figure_via_anchor(self, question: str, target_paper: Any, figures: list, context: Any) -> Any:
        if len(figures) == 1:
            return figures[0]
        try:
            anchor = await self.infer_cached_visual_anchor(
                papers=[target_paper],
                document_ids=[str(target_paper.metadata.get("document_id") or "")],
                question=question,
                load_cached_figure_payload=context.research_service._load_cached_figure_payload,
            )
        except Exception:
            logger.debug("infer_cached_visual_anchor failed, using fallback", exc_info=True)
            anchor = None
        if anchor is not None:
            anchor_figure_id = str(anchor.get("figure_id") or "").strip()
            if anchor_figure_id:
                matched = next((f for f in figures if f.figure_id == anchor_figure_id), None)
                if matched is not None:
                    logger.info("analyze_paper_figures: anchor selected figure_id=%s", anchor_figure_id)
                    return matched
        return figures[0]

    @staticmethod
    def _export_figure_image(analysis_response: Any, task_id: str) -> str | None:
        source_path = getattr(analysis_response.figure, "image_path", None) if analysis_response.figure else None
        if not source_path or not Path(source_path).is_file():
            return None
        export_dir = Path(".data/storage/figure_exports") / task_id
        export_dir.mkdir(parents=True, exist_ok=True)
        figure_id = getattr(analysis_response.figure, "figure_id", "") or "figure"
        safe_name = re.sub(r"[^\w\-_.]", "_", figure_id)
        suffix = Path(source_path).suffix or ".png"
        dest_path = export_dir / f"{safe_name}{suffix}"
        try:
            shutil.copy2(source_path, dest_path)
            logger.info("Exported figure image: %s -> %s", source_path, dest_path)
            return str(dest_path)
        except Exception:
            logger.debug("Failed to export figure image", exc_info=True)
            return str(source_path)

    # ------------------------------------------------------------------
    # Legacy unified runtime entry point (will be removed in Step 6)
    # ------------------------------------------------------------------

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
        supervisor_instruction: str | None = None,
    ) -> AnalyzeResearchPaperFigureResponse:
        analyze_target = load_cached_figure_target(paper=paper, figure_id=request.figure_id)
        if analyze_target is None:
            parsed_document = await parse_imported_paper_document(paper=paper, graph_runtime=graph_runtime)
            analyze_target = self.resolve_figure_target(
                paper=paper,
                parsed_document=parsed_document,
                request=request,
            )
        chart_ctx: dict[str, Any] = {
            "paper_id": paper.paper_id,
            "paper_title": paper.title,
            "figure_id": analyze_target.figure_id,
            "figure_source": analyze_target.source,
            "figure_title": analyze_target.metadata.get("title"),
            "figure_caption": analyze_target.metadata.get("caption"),
            "page_number": analyze_target.page_number,
            "bbox": analyze_target.bbox.model_dump(mode="json") if analyze_target.bbox else None,
            "selection_rationale": analyze_target.metadata.get("anchor_rationale"),
        }
        if supervisor_instruction:
            chart_ctx["supervisor_instruction"] = supervisor_instruction
        chart_result = await self.understand_chart(
            graph_runtime=graph_runtime,
            image_path=analyze_target.image_path,
            document_id=analyze_target.document_id,
            page_id=analyze_target.page_id,
            page_number=analyze_target.page_number,
            chart_id=analyze_target.chart_id,
            session_id=None,
            context=chart_ctx,
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
