"""Analyze paper figures supervisor tool."""

from __future__ import annotations

import logging
from pathlib import Path
import re
import shutil

from agents.chart_analysis_agent import ChartAnalysisAgent
from agents.research_supervisor_agent import ResearchSupervisorDecision
from domain.schemas.research import AnalyzeResearchPaperFigureRequest
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.unified_action_adapters import resolve_active_message

logger = logging.getLogger(__name__)


class AnalyzePaperFiguresTool:
    name = "analyze_paper_figures"

    def __init__(self, *, chart_analysis_agent: ChartAnalysisAgent) -> None:
        self.chart_analysis_agent = chart_analysis_agent

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
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
        if paper_ids:
            target_papers = [p for p in imported_papers if p.paper_id in paper_ids]
        else:
            target_papers = imported_papers
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
        if exported_image_path:
            if analysis_response.chart and hasattr(analysis_response.chart, "metadata"):
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


    @staticmethod
    def _export_figure_image(
        analysis_response,
        task_id: str,
    ) -> str | None:
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

    async def _select_figure_via_anchor(
        self,
        question: str,
        target_paper,
        figures: list,
        context: ResearchAgentToolContext,
    ):
        if len(figures) == 1:
            return figures[0]
        try:
            anchor = await self.chart_analysis_agent.infer_cached_visual_anchor(
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
