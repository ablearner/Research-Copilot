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
_VISION_FIGURE_VERIFY_MAX = 24
_NEGATIVE_FIGURE_FEEDBACK_RE = re.compile(
    r"(这张不是|不是这张|不是这个|不是这幅|不对|找错|错图|换一张|下一张|继续找|重新找|not\s+this|wrong\s+(figure|image|chart)|try\s+another)",
    re.IGNORECASE,
)


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
        question = str(payload.get("question") or payload.get("target_description") or context.request.message or "").strip()

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
        excluded_figure_ids = self._excluded_figure_ids_from_feedback(
            payload=payload,
            context=context,
            question=question,
            paper_id=target_paper.paper_id,
        )

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

        available_figures = [
            figure
            for figure in figure_list.figures
            if str(getattr(figure, "figure_id", "") or "").strip() not in excluded_figure_ids
        ]
        if excluded_figure_ids and not available_figures:
            return ResearchToolResult(
                status="skipped",
                observation=f"all discovered figures in paper '{target_paper.title}' were already rejected",
                metadata={
                    "reason": "no_untried_figures",
                    "paper_id": target_paper.paper_id,
                    "excluded_figure_ids": sorted(excluded_figure_ids),
                    "candidate_figure_count": len(figure_list.figures),
                    "observation_envelope": {
                        "progress_made": False,
                        "missing_inputs": ["figure_scope"],
                        "suggested_next_actions": ["clarify_request"],
                        "state_delta": {"rejected_figure_ids": sorted(excluded_figure_ids)},
                    },
                },
            )

        best_figure = await self._select_figure_via_anchor(
            question=question or "",
            target_paper=target_paper,
            figures=figure_list.figures,
            context=context,
            excluded_figure_ids=excluded_figure_ids,
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

        if isinstance(analysis_response.metadata, dict):
            analysis_response.metadata["paper_figure_rejected_ids"] = sorted(excluded_figure_ids)
            analysis_response.metadata["paper_figure_candidate_count"] = len(figure_list.figures)
            analysis_response.metadata["paper_figure_untried_count"] = len(available_figures)
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
                "excluded_figure_ids": sorted(excluded_figure_ids),
                "candidate_figure_count": len(figure_list.figures),
                "untried_figure_count": len(available_figures),
            },
        )

    async def _select_figure_via_anchor(
        self,
        question: str,
        target_paper: Any,
        figures: list,
        context: Any,
        excluded_figure_ids: set[str] | None = None,
    ) -> Any:
        excluded_ids = set(excluded_figure_ids or set())
        candidate_figures = [
            figure
            for figure in figures
            if str(getattr(figure, "figure_id", "") or "").strip() not in excluded_ids
        ]
        if not candidate_figures:
            candidate_figures = list(figures)
        if len(candidate_figures) == 1:
            return candidate_figures[0]
        try:
            anchor = await self.infer_cached_visual_anchor(
                papers=[target_paper],
                document_ids=[str(target_paper.metadata.get("document_id") or "")],
                question=question,
                load_cached_figure_payload=context.research_service._load_cached_figure_payload,
                exclude_figure_ids=excluded_ids,
            )
        except Exception:
            logger.debug("infer_cached_visual_anchor failed, using fallback", exc_info=True)
            anchor = None
        if anchor is not None:
            anchor_figure_id = str(anchor.get("figure_id") or "").strip()
            if anchor_figure_id:
                matched = next((f for f in candidate_figures if f.figure_id == anchor_figure_id), None)
                if matched is not None:
                    logger.info("analyze_paper_figures: anchor selected figure_id=%s", anchor_figure_id)
                    vision_match = await self._select_figure_via_vision(
                        question=question,
                        figures=self._prioritize_figures(matched, candidate_figures),
                        context=context,
                    )
                    return vision_match or matched
        fallback = candidate_figures[0]
        vision_match = await self._select_figure_via_vision(
            question=question,
            figures=candidate_figures,
            context=context,
        )
        return vision_match or fallback

    @staticmethod
    def _coerce_figure_id_set(value: Any) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            raw_items = re.split(r"[\s,，]+", value)
        elif isinstance(value, (list, tuple, set)):
            raw_items = list(value)
        else:
            raw_items = [value]
        return {str(item).strip() for item in raw_items if str(item).strip()}

    @staticmethod
    def _metadata_dict(value: Any) -> dict[str, Any]:
        return dict(value) if isinstance(value, dict) else {}

    def _excluded_figure_ids_from_feedback(
        self,
        *,
        payload: dict[str, Any],
        context: Any,
        question: str,
        paper_id: str,
    ) -> set[str]:
        excluded: set[str] = set()
        for key in ("exclude_figure_ids", "excluded_figure_ids", "rejected_figure_ids", "attempted_figure_ids"):
            excluded.update(self._coerce_figure_id_set(payload.get(key)))

        request_metadata = self._metadata_dict(getattr(getattr(context, "request", None), "metadata", None))
        for key in ("exclude_figure_ids", "excluded_figure_ids", "rejected_figure_ids", "attempted_figure_ids"):
            excluded.update(self._coerce_figure_id_set(request_metadata.get(key)))

        if not self._has_negative_figure_feedback(question):
            return self._filter_figure_ids_for_paper(excluded, paper_id)

        workspace_metadata = self._metadata_dict(getattr(getattr(context, "workspace", None), "metadata", None))
        feedback = self._metadata_dict(workspace_metadata.get("paper_figure_feedback"))
        excluded.update(self._coerce_figure_id_set(feedback.get("rejected_figure_ids")))
        excluded.update(self._coerce_figure_id_set(workspace_metadata.get("rejected_paper_figure_ids")))

        latest = self._metadata_dict(workspace_metadata.get("latest_paper_figure_analysis"))
        excluded.update(self._coerce_figure_id_set(latest.get("rejected_figure_ids")))
        excluded.update(self._coerce_figure_id_set(latest.get("figure_id")))
        excluded.update(self._coerce_figure_id_set(workspace_metadata.get("last_visual_anchor_figure_id")))

        report_metadata = self._metadata_dict(getattr(getattr(context, "report", None), "metadata", None))
        excluded.update(self._coerce_figure_id_set(report_metadata.get("last_visual_anchor_figure_id")))

        chart_result = getattr(context, "chart_result", None)
        figure = getattr(chart_result, "figure", None)
        excluded.update(self._coerce_figure_id_set(getattr(figure, "figure_id", None)))
        return self._filter_figure_ids_for_paper(excluded, paper_id)

    @staticmethod
    def _has_negative_figure_feedback(question: str) -> bool:
        return bool(_NEGATIVE_FIGURE_FEEDBACK_RE.search(question or ""))

    @staticmethod
    def _filter_figure_ids_for_paper(figure_ids: set[str], paper_id: str) -> set[str]:
        prefix = f"{paper_id}:"
        return {
            figure_id
            for figure_id in figure_ids
            if not paper_id or figure_id.startswith(prefix) or ":" not in figure_id
        }

    @staticmethod
    def _prioritize_figures(preferred: Any, figures: list[Any]) -> list[Any]:
        preferred_id = str(getattr(preferred, "figure_id", "") or "").strip()
        ordered = [preferred]
        ordered.extend(
            figure
            for figure in figures
            if str(getattr(figure, "figure_id", "") or "").strip() != preferred_id
        )
        return ordered

    async def _select_figure_via_vision(self, *, question: str, figures: list[Any], context: Any) -> Any | None:
        if not question.strip() or len(figures) <= 1:
            return None
        adapter = self._vision_adapter_from_context(context)
        if adapter is None or not hasattr(adapter, "analyze_image_structured"):
            return None
        from pydantic import BaseModel, Field

        class _FigureVisionMatch(BaseModel):
            matches: bool = False
            confidence: float = Field(default=0.0, ge=0.0, le=1.0)
            rationale: str = ""

        best_match: tuple[float, Any] | None = None
        for figure in figures[:_VISION_FIGURE_VERIFY_MAX]:
            image_path = str(getattr(figure, "image_path", "") or "").strip()
            if not image_path or not Path(image_path).is_file():
                continue
            prompt = (
                "你是科研论文图像定位器。判断这张候选图是否就是用户要找的图。\n\n"
                f"用户问题：{question.strip()}\n"
                f"候选 figure_id：{getattr(figure, 'figure_id', '')}\n"
                f"页码：{getattr(figure, 'page_number', None)}\n"
                f"标题：{getattr(figure, 'title', None) or ''}\n"
                f"图注：{getattr(figure, 'caption', None) or ''}\n\n"
                "只根据图像内容和这些上下文判断是否匹配用户要找的图。"
                "如果只是泛泛相关但不是目标图，matches=false。返回结构化字段。"
            )
            try:
                result = await adapter.analyze_image_structured(
                    prompt=prompt,
                    image_path=image_path,
                    response_model=_FigureVisionMatch,
                )
            except Exception:
                logger.debug("paper figure vision verification failed", exc_info=True)
                continue
            confidence = float(result.confidence)
            if result.matches and confidence >= 0.55:
                if best_match is None or confidence > best_match[0]:
                    best_match = (confidence, figure)
                if confidence >= 0.78:
                    break
        if best_match is None:
            return None
        logger.info(
            "analyze_paper_figures: vision selected figure_id=%s confidence=%.3f",
            getattr(best_match[1], "figure_id", None),
            best_match[0],
        )
        return best_match[1]

    @staticmethod
    def _vision_adapter_from_context(context: Any) -> Any | None:
        chart_tools = getattr(getattr(context, "graph_runtime", None), "chart_tools", None)
        adapter = getattr(chart_tools, "llm_adapter", None)
        if adapter is not None:
            return adapter
        return getattr(context, "llm_adapter", None)

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
        exclude_figure_ids: set[str] | None = None,
    ) -> dict[str, Any] | None:
        return await self.visual_anchor_skill.infer_cached_visual_anchor(
            papers=papers,
            document_ids=document_ids,
            question=question,
            load_cached_figure_payload=load_cached_figure_payload,
            exclude_figure_ids=exclude_figure_ids,
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
