"""QA routing and execution mixin for LiteratureResearchService.

Extracts all QA-related methods into a cohesive mixin so that the main
service file stays focused on task lifecycle and conversation management.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from agents.research_knowledge_agent import merge_retrieval_hits
from core.utils import now_iso as _now_iso
from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research import (
    PaperCandidate,
    ResearchReport,
    ResearchTask,
    ResearchTaskAskRequest,
    ResearchTaskAskResponse,
    ResearchTodoItem,
)
from services.research.research_context import ResearchExecutionContext
from services.research.research_workspace import build_workspace_from_task

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services.research.paper_selector_service import PaperSelectionScope

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchQARouteDecision:
    route: str
    confidence: float
    rationale: str
    visual_anchor: dict[str, Any] | None = None
    recovery_count: int = 0


class QARoutingMixin:
    """Mixin providing QA routing, execution, and follow-up for research tasks.

    Assumes the host class exposes the following attributes (provided by
    ``LiteratureResearchService.__init__``):

    * ``report_service``
    * ``paper_selector_service``
    * ``qa_routing_skill``
    * ``user_intent_resolver``
    * ``chart_analysis_agent``
    * ``memory_manager``
    * ``research_knowledge_agent``
    * ``research_writer_agent``
    * ``research_qa_runtime``
    * ``paper_search_service``
    * ``build_execution_context(...)``
    * ``save_task_state(...)``
    * ``_update_research_memory(...)``
    * ``_load_cached_figure_payload(...)``
    * ``list_paper_figures(...)``
    """

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def ask_task_collection(
        self,
        task_id: str,
        request: ResearchTaskAskRequest,
        *,
        graph_runtime,
    ) -> ResearchTaskAskResponse:
        task = self.report_service.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        report = self.report_service.load_report(task.task_id, task.report_id)
        papers = self.report_service.load_papers(task.task_id)
        scope = self.paper_selector_service.resolve_qa_scope(
            task=task,
            papers=papers,
            requested_paper_ids=request.paper_ids,
            requested_document_ids=request.document_ids,
        )
        if scope.explicit_scope and not scope.paper_ids and not scope.document_ids:
            raise ValueError("The requested paper/document scope did not match the current research task.")
        document_ids = list(scope.document_ids)
        scoped_papers = list(scope.papers or papers)
        if not document_ids and not scoped_papers and report is None:
            raise ValueError(f"Research task has no imported documents or persisted research artifacts available for QA: {task_id}")
        routing_authority = str(request.metadata.get("routing_authority") or "").strip()
        preferred_qa_route = str(request.metadata.get("preferred_qa_route") or "").strip()
        user_intent = None
        if routing_authority != "supervisor_llm":
            user_intent = await self.user_intent_resolver.resolve_async(
                message=request.question,
                has_task=True,
                candidate_paper_count=len(papers),
                candidate_papers=[
                    {
                        "paper_id": paper.paper_id,
                        "index": index,
                        "title": paper.title,
                        "source": paper.source,
                        "year": paper.year,
                    }
                    for index, paper in enumerate(papers, start=1)
                ],
                active_paper_ids=[
                    str(item).strip()
                    for item in ((request.metadata.get("context") or {}).get("active_paper_ids", []) if isinstance(request.metadata.get("context"), dict) else [])
                    if str(item).strip()
                ],
                selected_paper_ids=list(scope.paper_ids),
                has_visual_anchor=bool(request.image_path or request.chart_id or request.metadata.get("image_path")),
                has_document_input=False,
            )
            resolved_intent_paper_ids = [
                paper_id
                for paper_id in user_intent.resolved_paper_ids
                if paper_id in {paper.paper_id for paper in papers}
            ]
            if resolved_intent_paper_ids and not scope.paper_ids:
                scope = self.paper_selector_service.resolve_qa_scope(
                    task=task,
                    papers=papers,
                    requested_paper_ids=resolved_intent_paper_ids,
                    requested_document_ids=request.document_ids,
                )
                document_ids = list(scope.document_ids)
                scoped_papers = list(scope.papers or papers)
            if user_intent.needs_clarification:
                raise ValueError(user_intent.clarification_question or "当前问题指向不明确，请补充具体论文或图表。")
        if preferred_qa_route in {"collection_qa", "document_drilldown", "chart_drilldown"}:
            qa_route_decision = ResearchQARouteDecision(
                route=preferred_qa_route,  # type: ignore[arg-type]
                confidence=0.99,
                rationale="Supervisor selected the QA route explicitly.",
                visual_anchor=self._extract_visual_anchor(request=request, metadata=request.metadata),
            )
        else:
            qa_route_decision = await self._select_qa_route(
                question=request.question,
                scope_mode=scope.scope_mode,
                paper_ids=scope.paper_ids,
                document_ids=document_ids,
                request=request,
                metadata=request.metadata,
            )
        logger.info(
            "Research QA route selected: route=%s confidence=%.2f has_visual_anchor=%s request_image_path=%s request_chart_id=%s request_page_id=%s request_page_number=%s",
            qa_route_decision.route,
            qa_route_decision.confidence,
            qa_route_decision.visual_anchor is not None,
            bool(str(request.image_path or "").strip()),
            str(request.chart_id or ""),
            str(request.page_id or ""),
            request.page_number,
        )
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is None:
            inferred_visual_anchor = await self._infer_or_discover_visual_anchor(
                task_id=task.task_id,
                papers=scoped_papers,
                document_ids=document_ids,
                question=request.question,
                graph_runtime=graph_runtime,
            )
            if inferred_visual_anchor is not None:
                logger.info(
                    "Research QA inferred visual anchor for chart drilldown: image_path=%s chart_id=%s page_id=%s page_number=%s",
                    str(inferred_visual_anchor.get("image_path") or ""),
                    str(inferred_visual_anchor.get("chart_id") or ""),
                    str(inferred_visual_anchor.get("page_id") or ""),
                    inferred_visual_anchor.get("page_number"),
                )
                qa_route_decision = ResearchQARouteDecision(
                    route=qa_route_decision.route,
                    confidence=max(qa_route_decision.confidence, 0.9),
                    rationale=(
                        f"{qa_route_decision.rationale} Auto-selected a paper figure preview "
                        "to ground chart-focused QA."
                    ),
                    visual_anchor=inferred_visual_anchor,
                )
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is None:
            restored_visual_anchor = self._restore_visual_anchor_from_workspace(task=task, report=report)
            if restored_visual_anchor is not None:
                logger.info(
                    "Research QA restored visual anchor from workspace: image_path=%s chart_id=%s page_id=%s page_number=%s",
                    str(restored_visual_anchor.get("image_path") or ""),
                    str(restored_visual_anchor.get("chart_id") or ""),
                    str(restored_visual_anchor.get("page_id") or ""),
                    restored_visual_anchor.get("page_number"),
                )
                qa_route_decision = ResearchQARouteDecision(
                    route=qa_route_decision.route,
                    confidence=max(qa_route_decision.confidence, 0.9),
                    rationale=(
                        f"{qa_route_decision.rationale} Restored the latest visual anchor from the saved workspace "
                        "to ground chart-focused QA."
                    ),
                    visual_anchor=restored_visual_anchor,
                )
        scoped_request = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "qa_route": qa_route_decision.route,
                    "qa_route_confidence": qa_route_decision.confidence,
                    "qa_route_rationale": qa_route_decision.rationale,
                    "visual_anchor": qa_route_decision.visual_anchor,
                    "qa_scope_mode": scope.scope_mode,
                    "selected_paper_ids": scope.paper_ids,
                    "selected_document_ids": document_ids,
                    "selected_paper_titles": scope.selected_titles(),
                    "selection_warnings": scope.warnings,
                    "selection_summary": scope.metadata.get("selection_summary"),
                    "scope_metadata": scope.metadata,
                    "user_intent": user_intent.model_dump(mode="json") if user_intent is not None else None,
                    "routing_authority": routing_authority or None,
                    "preferred_qa_route": preferred_qa_route or None,
                }
            }
        )
        execution_context = self.build_execution_context(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            task=task,
            report=report,
            papers=scoped_papers,
            document_ids=document_ids,
            selected_paper_ids=scope.paper_ids,
            skill_name=request.skill_name,
            reasoning_style=request.reasoning_style,
            metadata=scoped_request.metadata,
        )
        qa = await self._run_scoped_task_qa(
            graph_runtime=graph_runtime,
            task=task,
            request=scoped_request,
            report=report,
            papers=scoped_papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa_route_decision=qa_route_decision,
        )
        quality_check = self._build_answer_quality_check(
            qa=qa,
            route=qa_route_decision.route,
            scope_mode=scope.scope_mode,
            document_ids=document_ids,
        )
        qa, qa_route_decision, quality_check = await self._maybe_recover_qa_route(
            graph_runtime=graph_runtime,
            task=task,
            request=scoped_request,
            report=report,
            papers=scoped_papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa=qa,
            qa_route_decision=qa_route_decision,
            quality_check=quality_check,
            scope=scope,
        )
        qa = qa.model_copy(
            update={
                "metadata": {
                    **qa.metadata,
                    **request.metadata,
                    "qa_route": qa_route_decision.route,
                    "qa_route_confidence": qa_route_decision.confidence,
                    "qa_route_rationale": qa_route_decision.rationale,
                    "visual_anchor": qa_route_decision.visual_anchor,
                    "research_task_id": task_id,
                    "research_topic": task.topic,
                    "qa_scope_mode": scope.scope_mode,
                    "selected_paper_ids": scope.paper_ids,
                    "selected_document_ids": document_ids,
                    "selected_paper_titles": scope.selected_titles(),
                    "selection_warnings": scope.warnings,
                    "selection_summary": scope.metadata.get("selection_summary"),
                    "scope_metadata": scope.metadata,
                    "qa_route_recovery_count": qa_route_decision.recovery_count,
                }
            }
        )
        qa_metadata = qa.metadata if isinstance(qa.metadata, dict) else {}
        visual_anchor_figure = self.chart_analysis_agent.resolve_visual_anchor_figure(
            papers=scoped_papers,
            qa_metadata=qa_metadata,
            load_cached_figure_payload=self._load_cached_figure_payload,
        )
        if visual_anchor_figure is not None:
            qa = qa.model_copy(
                update={
                    "metadata": {
                        **qa_metadata,
                        "visual_anchor_figure": visual_anchor_figure.model_dump(mode="json"),
                    }
                }
            )
        qa = qa.model_copy(
            update={
                "metadata": {
                    **(qa.metadata if isinstance(qa.metadata, dict) else {}),
                    "answer_quality_check": quality_check,
                }
            }
        )
        updated_task, updated_report = self._apply_qa_follow_up(
            task=task,
            request=scoped_request,
            qa=qa,
            papers=papers,
            document_ids=document_ids,
            scope=scope,
        )
        self.save_task_state(updated_task, conversation_id=request.conversation_id)
        self.report_service.save_report(updated_report)
        self._update_research_memory(
            graph_runtime=graph_runtime,
            conversation_id=request.conversation_id,
            task=updated_task,
            report=updated_report,
            papers=scoped_papers,
            document_ids=document_ids,
            selected_paper_ids=scope.paper_ids,
            task_intent=f"research_{qa_route_decision.route}:{scope.scope_mode}",
            question=request.question,
            answer=qa.answer,
            retrieval_summary=(
                f"documents={len(document_ids)}, evidences={len(qa.evidence_bundle.evidences)}, "
                f"confidence={qa.confidence if qa.confidence is not None else 'empty'}"
            ),
            metadata_update={
                "last_skill_name": request.skill_name,
                "reasoning_style": request.reasoning_style or "cot",
                "evidence_count": len(qa.evidence_bundle.evidences),
                "qa_scope_mode": scope.scope_mode,
                "selected_paper_ids": scope.paper_ids[:8],
                "answer_quality_check": quality_check,
            },
        )
        if not quality_check["needs_recovery"] and execution_context.session_id:
            self.memory_manager.promote_conclusion_to_long_term(
                execution_context.session_id,
                conclusion=self._compact_text(qa.answer, limit=700),
                topic=task.topic,
                keywords=[qa_route_decision.route, scope.scope_mode],
                related_paper_ids=scope.paper_ids[:8],
                metadata={
                    "question": request.question,
                    "confidence": qa.confidence,
                    "evidence_count": len(qa.evidence_bundle.evidences),
                },
            )
        return ResearchTaskAskResponse(
            task_id=task_id,
            paper_ids=scope.paper_ids,
            document_ids=document_ids,
            scope_mode=scope.scope_mode,
            qa=qa,
            report=updated_report,
            todo_items=updated_task.todo_items,
            warnings=scope.warnings,
        )

    # ------------------------------------------------------------------
    # Route selection
    # ------------------------------------------------------------------

    async def _select_qa_route(
        self,
        *,
        question: str,
        scope_mode: str,
        paper_ids: list[str],
        document_ids: list[str],
        request: ResearchTaskAskRequest,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchQARouteDecision:
        visual_anchor = self._extract_visual_anchor(request=request, metadata=metadata)
        route_result = await self.qa_routing_skill.classify_async(
            question=question,
            scope_mode=scope_mode,
            paper_ids=paper_ids,
            document_ids=document_ids,
            has_visual_anchor=visual_anchor is not None,
        )
        return ResearchQARouteDecision(
            route=route_result.route,
            confidence=route_result.confidence,
            rationale=route_result.rationale,
            visual_anchor=visual_anchor if route_result.route == "chart_drilldown" else None,
        )

    # ------------------------------------------------------------------
    # Visual anchor helpers
    # ------------------------------------------------------------------

    def _restore_visual_anchor_from_workspace(
        self,
        *,
        task: ResearchTask,
        report: ResearchReport | None,
    ) -> dict[str, Any] | None:
        workspace_candidates = [
            task.workspace.metadata if isinstance(task.workspace.metadata, dict) else {},
            report.workspace.metadata if report is not None and isinstance(report.workspace.metadata, dict) else {},
        ]
        for metadata in workspace_candidates:
            anchor = metadata.get("last_visual_anchor")
            if not isinstance(anchor, dict):
                continue
            image_path = str(anchor.get("image_path") or "").strip()
            if not image_path:
                continue
            restored: dict[str, Any] = {"image_path": image_path}
            page_id = str(anchor.get("page_id") or "").strip()
            if page_id:
                restored["page_id"] = page_id
            chart_id = str(anchor.get("chart_id") or "").strip()
            if chart_id:
                restored["chart_id"] = chart_id
            raw_page_number = anchor.get("page_number")
            try:
                page_number = int(raw_page_number) if raw_page_number is not None else None
            except (TypeError, ValueError):
                page_number = None
            if page_number is not None and page_number >= 1:
                restored["page_number"] = page_number
            for key in ("figure_id", "anchor_source", "anchor_selection", "anchor_rationale"):
                value = anchor.get(key)
                if isinstance(value, str) and value.strip():
                    restored[key] = value.strip()
            return restored
        return None

    def _extract_visual_anchor(
        self,
        *,
        request: ResearchTaskAskRequest,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        image_path = str(request.image_path or (metadata or {}).get("image_path") or "").strip()
        if not image_path:
            return None
        anchor = {"image_path": image_path}
        page_id = request.page_id or (metadata or {}).get("page_id")
        if page_id:
            anchor["page_id"] = str(page_id)
        try:
            raw_page_number = (
                request.page_number
                if request.page_number is not None
                else (metadata or {}).get("page_number")
            )
            page_number = int(raw_page_number) if raw_page_number is not None else None
        except (TypeError, ValueError):
            page_number = None
        if page_number is not None and page_number >= 1:
            anchor["page_number"] = page_number
        chart_id = request.chart_id or (metadata or {}).get("chart_id")
        if chart_id:
            anchor["chart_id"] = str(chart_id)
        return anchor

    async def _infer_cached_visual_anchor(
        self,
        *,
        papers: list[PaperCandidate],
        document_ids: list[str],
        question: str,
    ) -> dict[str, Any] | None:
        return await self.chart_analysis_agent.infer_cached_visual_anchor(
            papers=papers,
            document_ids=document_ids,
            question=question,
            load_cached_figure_payload=self._load_cached_figure_payload,
        )

    async def _infer_or_discover_visual_anchor(
        self,
        *,
        task_id: str,
        papers: list[PaperCandidate],
        document_ids: list[str],
        question: str,
        graph_runtime: Any,
    ) -> dict[str, Any] | None:
        inferred = await self._infer_cached_visual_anchor(
            papers=papers,
            document_ids=document_ids,
            question=question,
        )
        if inferred is not None:
            return inferred
        discovered = await self._discover_figures_for_scope(
            task_id=task_id,
            papers=papers,
            document_ids=document_ids,
            graph_runtime=graph_runtime,
        )
        if not discovered:
            return None
        refreshed_papers = self.report_service.load_papers(task_id)
        scoped_papers = [
            paper for paper in refreshed_papers
            if not papers or paper.paper_id in {item.paper_id for item in papers}
        ] or refreshed_papers
        return await self._infer_cached_visual_anchor(
            papers=scoped_papers,
            document_ids=document_ids,
            question=question,
        )

    async def _discover_figures_for_scope(
        self,
        *,
        task_id: str,
        papers: list[PaperCandidate],
        document_ids: list[str],
        graph_runtime: Any,
    ) -> bool:
        allowed_document_ids = {item for item in document_ids if item}
        discovered = False
        for paper in papers:
            document_id = str(paper.metadata.get("document_id") or "").strip()
            storage_uri = str(paper.metadata.get("storage_uri") or "").strip()
            if paper.ingest_status != "ingested" or not document_id or not storage_uri:
                continue
            if allowed_document_ids and document_id not in allowed_document_ids:
                continue
            if self._load_cached_figure_payload(paper=paper) is not None:
                discovered = True
                continue
            try:
                await self.list_paper_figures(
                    task_id,
                    paper.paper_id,
                    graph_runtime=graph_runtime,
                )
                discovered = True
            except Exception:
                continue
        return discovered

    # ------------------------------------------------------------------
    # QA execution
    # ------------------------------------------------------------------

    async def _run_scoped_task_qa(
        self,
        *,
        graph_runtime,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
        qa_route_decision: ResearchQARouteDecision,
    ) -> QAResponse:
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is None and not document_ids:
            selected_titles = [
                str(item).strip()
                for item in (request.metadata.get("selected_paper_titles", []) if isinstance(request.metadata, dict) else [])
                if str(item).strip()
            ]
            scoped_target = selected_titles[0] if len(selected_titles) == 1 else "当前目标论文"
            return QAResponse(
                answer=(
                    f"我知道你是在问 {scoped_target} 里的图表/系统框图，但当前工作区还没有这篇论文的已导入正文或图表锚点，"
                    "所以我不能可靠解释具体图示内容。请先导入这篇论文，或明确指定图号/上传图像后再问。"
                ),
                question=request.question,
                evidence_bundle=EvidenceBundle(
                    summary="chart_question_without_document_scope",
                    metadata={
                        "reason": "chart_question_without_document_scope",
                        "qa_route": qa_route_decision.route,
                    },
                ),
                confidence=0.18,
                metadata={
                    **(request.metadata if isinstance(request.metadata, dict) else {}),
                    "qa_route": qa_route_decision.route,
                    "qa_route_source": "literature_research_service",
                    "chart_drilldown_blocked_reason": "missing_document_scope_and_visual_anchor",
                },
            )
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is not None:
            handle_ask_fused = getattr(graph_runtime, "handle_ask_fused", None)
            visual_anchor_figure = self.chart_analysis_agent.resolve_visual_anchor_figure(
                papers=papers,
                qa_metadata={"visual_anchor": qa_route_decision.visual_anchor},
                load_cached_figure_payload=self._load_cached_figure_payload,
            )
            figure_context = (
                {
                    "figure": visual_anchor_figure.model_dump(mode="json"),
                    "paper_id": visual_anchor_figure.paper_id,
                    "figure_id": visual_anchor_figure.figure_id,
                    "chart_id": visual_anchor_figure.chart_id,
                    "page_id": visual_anchor_figure.page_id,
                    "page_number": visual_anchor_figure.page_number,
                    "title": visual_anchor_figure.title,
                    "caption": visual_anchor_figure.caption,
                    "source": visual_anchor_figure.source,
                }
                if visual_anchor_figure is not None
                else {}
            )
            logger.info(
                "Research QA attempting fused chart drilldown: has_handle_ask_fused=%s image_path=%s chart_id=%s page_id=%s page_number=%s document_count=%s",
                callable(handle_ask_fused),
                str(qa_route_decision.visual_anchor.get("image_path") or ""),
                str(qa_route_decision.visual_anchor.get("chart_id") or ""),
                str(qa_route_decision.visual_anchor.get("page_id") or ""),
                qa_route_decision.visual_anchor.get("page_number"),
                len(document_ids),
            )
            if callable(handle_ask_fused):
                fused_result = await handle_ask_fused(
                    question=request.question,
                    doc_id=document_ids[0] if len(document_ids) == 1 else None,
                    document_ids=document_ids,
                    top_k=request.top_k,
                    session_id=execution_context.session_id,
                    filters={
                        "research_task_id": task.task_id,
                        "research_topic": task.topic,
                        "qa_mode": "research_collection",
                        "qa_route": qa_route_decision.route,
                        "qa_scope_mode": request.metadata.get("qa_scope_mode"),
                        "selected_paper_ids": request.metadata.get("selected_paper_ids", []),
                        "selected_document_ids": request.metadata.get("selected_document_ids", []),
                    },
                    metadata={
                        **request.metadata,
                        "qa_route": qa_route_decision.route,
                        "qa_route_source": "literature_research_service",
                        "visual_anchor": qa_route_decision.visual_anchor,
                        "visual_anchor_figure": (
                            visual_anchor_figure.model_dump(mode="json")
                            if visual_anchor_figure is not None
                            else None
                        ),
                        "figure_context": figure_context,
                    },
                    skill_name=request.skill_name,
                    reasoning_style=request.reasoning_style,
                    **qa_route_decision.visual_anchor,
                )
                qa = fused_result.qa
                return qa.model_copy(
                    update={
                        "metadata": {
                            **qa.metadata,
                            "autonomy_mode": "task_scoped_drilldown",
                            "agent_architecture": "research_service_to_graph_runtime",
                            "primary_agents": ["RagRuntime"],
                            "selected_skill": request.skill_name,
                            "memory_enabled": execution_context.memory_enabled,
                            "session_id": execution_context.session_id,
                            "drilldown_runtime": "fused_chart",
                        }
                    }
                )
            logger.warning("Research QA chart drilldown fell back because graph_runtime.handle_ask_fused is not callable")

        if qa_route_decision.route in {"document_drilldown", "chart_drilldown"} and document_ids:
            handle_ask_document = getattr(graph_runtime, "handle_ask_document", None)
            if callable(handle_ask_document):
                logger.info(
                    "Research QA using document drilldown fallback: route=%s document_count=%s has_visual_anchor=%s",
                    qa_route_decision.route,
                    len(document_ids),
                    qa_route_decision.visual_anchor is not None,
                )
                qa = await handle_ask_document(
                    question=request.question,
                    document_ids=document_ids,
                    top_k=request.top_k,
                    filters={
                        "research_task_id": task.task_id,
                        "research_topic": task.topic,
                        "qa_mode": "research_collection",
                        "qa_route": qa_route_decision.route,
                        "qa_scope_mode": request.metadata.get("qa_scope_mode"),
                        "selected_paper_ids": request.metadata.get("selected_paper_ids", []),
                        "selected_document_ids": request.metadata.get("selected_document_ids", []),
                    },
                    session_id=execution_context.session_id,
                    task_intent=f"research_{qa_route_decision.route}",
                    metadata={
                        **request.metadata,
                        "qa_route": qa_route_decision.route,
                        "qa_route_source": "literature_research_service",
                    },
                    skill_name=request.skill_name,
                    reasoning_style=request.reasoning_style,
                )
                return qa.model_copy(
                    update={
                        "metadata": {
                            **qa.metadata,
                            "autonomy_mode": "task_scoped_drilldown",
                            "agent_architecture": "research_service_to_graph_runtime",
                            "primary_agents": ["RagRuntime"],
                            "selected_skill": request.skill_name,
                            "memory_enabled": execution_context.memory_enabled,
                            "session_id": execution_context.session_id,
                            "drilldown_runtime": "document",
                        }
                    }
                )

        return await self._run_direct_collection_qa(
            graph_runtime=graph_runtime,
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
        )

    async def _run_direct_collection_qa(
        self,
        *,
        graph_runtime,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
    ) -> QAResponse:
        if getattr(graph_runtime, "answer_tools", None) is None and self.research_qa_runtime is not None:
            legacy_result = await self.research_qa_runtime.run(
                graph_runtime=graph_runtime,
                task=task,
                request=request,
                report=report,
                papers=papers,
                document_ids=document_ids,
                execution_context=execution_context,
            )
            qa = legacy_result.qa
            return qa.model_copy(
                update={
                    "metadata": {
                        **(qa.metadata if isinstance(qa.metadata, dict) else {}),
                        "autonomy_mode": "lead_agent_loop",
                        "agent_architecture": "main_agents_only",
                        "qa_execution_path": "research_supervisor_legacy_fallback",
                        "memory_enabled": execution_context.memory_enabled,
                        "session_id": execution_context.session_id,
                    }
                }
            )

        resolver = getattr(graph_runtime, "resolve_skill_context", None)
        skill_context = (
            resolver(
                task_type="ask_document",
                preferred_skill_name=request.skill_name or "research_report",
            )
            if callable(resolver)
            else None
        )
        resolved_question = self._rewrite_collection_question(
            question=request.question,
            task=task,
            papers=papers,
            scope_mode=str((request.metadata or {}).get("qa_scope_mode") or "all_imported"),
        )
        request_with_resolution = request.model_copy(
            update={
                "metadata": {
                    **request.metadata,
                    "resolved_question": resolved_question,
                }
            }
        )
        runtime_state = SimpleNamespace(
            task=task,
            request=request_with_resolution,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            skill_context=skill_context,
            queries=[],
            completed_queries=set(),
            refinement_used=False,
            summary_checked=False,
            manifest_built=False,
            retrieval_hits=[],
            summary_hits=[],
            manifest_hits=[],
            evidence_bundle=EvidenceBundle(),
            retrieval_result=None,
            qa=None,
            warnings=[],
            trace=[],
            question=resolved_question,
            original_question=request.question,
            top_k=request.top_k,
        )
        runtime_state.queries = await self.research_knowledge_agent.plan_collection_queries(runtime_state)
        for query in list(runtime_state.queries):
            try:
                hits = await self.research_knowledge_agent.retrieve_collection_evidence(
                    graph_runtime=graph_runtime,
                    state=runtime_state,
                    query=query,
                )
                runtime_state.retrieval_hits = merge_retrieval_hits(
                    runtime_state.retrieval_hits,
                    hits,
                )
            except Exception as exc:  # pragma: no cover - provider/runtime failures are environment-specific
                runtime_state.warnings.append(f"collection_retrieval:{query} failed: {exc}")
            runtime_state.completed_queries.add(query)
        runtime_state.summary_checked = True
        try:
            summary_hits = await self.research_knowledge_agent.retrieve_graph_summary(
                graph_runtime=graph_runtime,
                state=runtime_state,
            )
            runtime_state.summary_hits = merge_retrieval_hits(
                runtime_state.summary_hits,
                summary_hits,
            )
        except Exception as exc:  # pragma: no cover - provider/runtime failures are environment-specific
            runtime_state.warnings.append(f"graph_summary:{runtime_state.question} failed: {exc}")
        runtime_state.manifest_hits = self.research_knowledge_agent.build_collection_manifest(runtime_state)
        runtime_state.manifest_built = True
        qa = await self.research_writer_agent.answer_collection_question(
            graph_runtime=graph_runtime,
            state=runtime_state,
            primary_agents=[
                "ResearchSupervisorAgent",
                "ResearchKnowledgeAgent",
                "ResearchWriterAgent",
            ],
        )
        return qa.model_copy(
            update={
                "metadata": {
                    **(qa.metadata if isinstance(qa.metadata, dict) else {}),
                    "autonomy_mode": "lead_agent_loop",
                    "agent_architecture": "main_agents_only",
                    "primary_agents": [
                        "ResearchSupervisorAgent",
                        "ResearchKnowledgeAgent",
                        "ResearchWriterAgent",
                    ],
                    "supervisor_execution_mode": "single_supervisor_action",
                    "supervisor_agent_architecture": "supervisor_direct_execution",
                    "qa_execution_path": "research_supervisor_direct",
                    "qa_warnings": list(runtime_state.warnings),
                    "planned_queries": list(runtime_state.queries),
                    "completed_queries": list(runtime_state.completed_queries),
                    "memory_enabled": execution_context.memory_enabled,
                    "session_id": execution_context.session_id,
                }
            }
        )

    def _rewrite_collection_question(
        self,
        *,
        question: str,
        task: ResearchTask,
        papers: list[PaperCandidate],
        scope_mode: str,
    ) -> str:
        normalized = str(question or "").strip()
        if not normalized:
            return normalized
        compact = re.sub(r"\s+", "", normalized.lower())
        if compact in {"效果怎么样", "效果如何", "表现怎么样", "表现如何"}:
            return (
                f"请结合研究主题“{task.topic}”对当前研究集合做综合评价，"
                "说明整体效果、证据强弱与主要边界，不要只回答单篇论文。"
            )
        if scope_mode == "all_imported":
            return normalized
        return normalized

    def _refresh_existing_pool(
        self,
        *,
        existing_papers: list[PaperCandidate],
        incoming_papers: list[PaperCandidate],
        ranking_topic: str,
    ) -> list[PaperCandidate]:
        merged = self.paper_search_service._dedupe([*existing_papers, *incoming_papers])
        return self.paper_search_service.paper_ranker.rank(
            topic=ranking_topic,
            papers=merged,
            max_papers=max(len(merged), 1),
        )

    # ------------------------------------------------------------------
    # Recovery & quality
    # ------------------------------------------------------------------

    async def _maybe_recover_qa_route(
        self,
        *,
        graph_runtime,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
        qa: QAResponse,
        qa_route_decision: ResearchQARouteDecision,
        quality_check: dict[str, Any],
        scope: PaperSelectionScope,
    ) -> tuple[QAResponse, ResearchQARouteDecision, dict[str, Any]]:
        rerouted = self._select_recovery_qa_route(
            request=request,
            scope=scope,
            document_ids=document_ids,
            qa=qa,
            qa_route_decision=qa_route_decision,
            quality_check=quality_check,
        )
        if rerouted is None:
            return qa, qa_route_decision, quality_check
        recovered_qa = await self._run_scoped_task_qa(
            graph_runtime=graph_runtime,
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa_route_decision=rerouted,
        )
        recovered_quality = self._build_answer_quality_check(
            qa=recovered_qa,
            route=rerouted.route,
            scope_mode=scope.scope_mode,
            document_ids=document_ids,
        )
        recovered_qa = recovered_qa.model_copy(
            update={
                "metadata": {
                    **(recovered_qa.metadata if isinstance(recovered_qa.metadata, dict) else {}),
                    "qa_route_recovered_from": qa_route_decision.route,
                    "qa_route_recovery_reason": rerouted.rationale,
                }
            }
        )
        return recovered_qa, rerouted, recovered_quality

    def _select_recovery_qa_route(
        self,
        *,
        request: ResearchTaskAskRequest,
        scope: PaperSelectionScope,
        document_ids: list[str],
        qa: QAResponse,
        qa_route_decision: ResearchQARouteDecision,
        quality_check: dict[str, Any],
    ) -> ResearchQARouteDecision | None:
        if qa_route_decision.recovery_count >= 1:
            return None
        if qa_route_decision.visual_anchor is not None:
            return None
        if not quality_check.get("needs_recovery"):
            return None
        if document_ids and qa_route_decision.route == "collection_qa" and scope.scope_mode in {"selected_documents", "selected_papers"}:
            return ResearchQARouteDecision(
                route="document_drilldown",
                confidence=max(qa_route_decision.confidence, 0.72),
                rationale=(
                    "The initial collection QA answer was under-supported for a narrowed paper/document scope, "
                    "so a single conservative retry uses document drilldown."
                ),
                visual_anchor=None,
                recovery_count=qa_route_decision.recovery_count + 1,
            )
        if qa_route_decision.route == "document_drilldown" and not document_ids:
            return ResearchQARouteDecision(
                route="collection_qa",
                confidence=max(qa_route_decision.confidence, 0.7),
                rationale=(
                    "The initial document drilldown route had no usable document scope, "
                    "so a single conservative retry broadens to collection QA."
                ),
                visual_anchor=None,
                recovery_count=qa_route_decision.recovery_count + 1,
            )
        return None

    # ------------------------------------------------------------------
    # Follow-up & quality helpers
    # ------------------------------------------------------------------

    def _apply_qa_follow_up(
        self,
        *,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        qa,
        papers: list[PaperCandidate],
        document_ids: list[str],
        scope: PaperSelectionScope,
    ) -> tuple[ResearchTask, ResearchReport]:
        now = datetime.now(UTC).isoformat()
        report = self.report_service.load_report(task.task_id, task.report_id)
        if report is None:
            report = ResearchReport(
                report_id=task.report_id or f"report_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
                task_id=task.task_id,
                topic=task.topic,
                generated_at=now,
                markdown=f"# 文献调研报告：{task.topic}",
                paper_count=task.paper_count,
                source_counts={},
                metadata={"writer": "qa_follow_up"},
            )

        evidence_count = len(qa.evidence_bundle.evidences)
        confidence_value = qa.confidence if qa.confidence is not None else 0.0
        insufficient = self._is_insufficient_answer(
            answer=qa.answer,
            confidence=confidence_value,
            evidence_count=evidence_count,
        )
        question_summary = self._compact_text(request.question, limit=120)
        answer_summary = self._compact_text(qa.answer, limit=220)

        todo_items = self._upsert_follow_up_todo(
            existing_items=task.todo_items,
            question=request.question,
            question_summary=question_summary,
            answer_summary=answer_summary,
            insufficient=insufficient,
            evidence_count=evidence_count,
            confidence=confidence_value,
            created_at=now,
        )
        latest_todo = todo_items[0] if todo_items else None
        updated_markdown = self._append_qa_report_entry(
            markdown=report.markdown,
            asked_at=now,
            question=request.question,
            answer=qa.answer,
            document_count=len(document_ids),
            evidence_count=evidence_count,
            confidence=qa.confidence,
            todo_item=latest_todo,
            paper_titles=scope.selected_titles(),
            scope_mode=scope.scope_mode,
        )

        updated_highlights = list(report.highlights)
        updated_gaps = list(report.gaps)
        qa_metadata = qa.metadata if isinstance(qa.metadata, dict) else {}
        visual_anchor = qa_metadata.get("visual_anchor") if isinstance(qa_metadata.get("visual_anchor"), dict) else None
        visual_anchor_figure = self.chart_analysis_agent.resolve_visual_anchor_figure(
            papers=papers,
            qa_metadata=qa_metadata,
            load_cached_figure_payload=self._load_cached_figure_payload,
        )
        if insufficient:
            gap = (
                f"关于“{question_summary}”的研究集合证据仍不足，当前仅有 {evidence_count} 条证据，"
                "建议补充更直接的论文或扩大检索范围。"
            )
            updated_gaps = self._prepend_unique_text(updated_gaps, gap)
        else:
            highlight = f"问答补充：关于“{question_summary}”，当前结论是 {answer_summary}"
            updated_highlights = self._prepend_unique_text(updated_highlights, highlight)

        report_metadata = dict(report.metadata)
        report_metadata.update(
            {
                "last_qa_at": now,
                "last_qa_question": question_summary,
                "qa_update_count": int(report.metadata.get("qa_update_count") or 0) + 1,
                "last_visual_anchor": visual_anchor,
                "last_visual_anchor_figure_id": (
                    visual_anchor_figure.figure_id if visual_anchor_figure is not None else None
                ),
            }
        )
        updated_report = report.model_copy(
            update={
                "generated_at": now,
                "markdown": updated_markdown,
                "highlights": updated_highlights[:10],
                "gaps": updated_gaps[:10],
                "metadata": report_metadata,
            }
        )
        updated_task = task.model_copy(
            update={
                "updated_at": now,
                "report_id": updated_report.report_id,
                "todo_items": todo_items[:20],
            }
        )
        workspace = build_workspace_from_task(
            task=updated_task,
            report=updated_report,
            papers=papers,
            stage="qa",
            extra_questions=[request.question],
            extra_findings=[qa.answer],
            stop_reason=(
                "Collection QA found an evidence gap that should drive the next retrieval cycle."
                if insufficient
                else "Collection QA completed and committed its answer back into the research workspace."
            ),
            metadata={
                "last_qa_question": question_summary,
                "last_qa_confidence": round(confidence_value, 4),
                "last_qa_evidence_count": evidence_count,
                "last_visual_anchor": visual_anchor,
                "last_visual_anchor_figure": (
                    visual_anchor_figure.model_dump(mode="json") if visual_anchor_figure is not None else None
                ),
                "last_visual_anchor_figure_id": (
                    visual_anchor_figure.figure_id if visual_anchor_figure is not None else None
                ),
            },
        )
        updated_report = updated_report.model_copy(update={"workspace": workspace})
        updated_task = updated_task.model_copy(update={"workspace": workspace})
        return updated_task, updated_report

    def _upsert_follow_up_todo(
        self,
        *,
        existing_items: list[ResearchTodoItem],
        question: str,
        question_summary: str,
        answer_summary: str,
        insufficient: bool,
        evidence_count: int,
        confidence: float,
        created_at: str,
    ) -> list[ResearchTodoItem]:
        if insufficient:
            content = f"补充与“{question_summary}”直接相关的论文，并在扩展关键词或时间窗口后重新运行研究任务。"
            rationale = f"当前仅检索到 {evidence_count} 条可用证据，置信度 {confidence:.2f}，现有研究集合无法稳定回答该问题。"
            source = "evidence_gap"
            priority = "high"
        else:
            content = f"围绕“{question_summary}”整理一个更细粒度的对比表，并持续核验新论文是否改变当前结论。"
            rationale = f"当前已有可用答案，可继续沉淀成综述结论与实验对比材料。摘要：{answer_summary}"
            source = "qa_follow_up"
            priority = "medium"

        next_item = ResearchTodoItem(
            todo_id=f"todo_{uuid4().hex}",
            content=content,
            rationale=rationale,
            status="open",
            priority=priority,
            created_at=created_at,
            question=question,
            source=source,
            metadata={
                "evidence_count": evidence_count,
                "confidence": round(confidence, 4),
            },
        )
        updated_items: list[ResearchTodoItem] = []
        replaced = False
        for item in existing_items:
            if item.question == question and item.status == "open":
                updated_items.append(next_item)
                replaced = True
            else:
                updated_items.append(item)
        if not replaced:
            updated_items.insert(0, next_item)
        return updated_items

    def _append_qa_report_entry(
        self,
        *,
        markdown: str,
        asked_at: str,
        question: str,
        answer: str,
        document_count: int,
        evidence_count: int,
        confidence: float | None,
        todo_item: ResearchTodoItem | None,
        paper_titles: list[str] | None = None,
        scope_mode: str | None = None,
    ) -> str:
        section_heading = "## 研究集合问答补充"
        entry_lines = [
            f"### {asked_at}",
            f"问题：{question}",
        ]
        entry_lines.extend(
            [
                "",
                "回答：",
                answer.strip() or "（空）",
            ]
        )
        if section_heading in markdown:
            return f"{markdown.rstrip()}\n\n" + "\n".join(entry_lines)
        prefix = markdown.rstrip()
        spacer = "\n\n" if prefix else ""
        return f"{prefix}{spacer}{section_heading}\n\n" + "\n".join(entry_lines)

    def _prepend_unique_text(self, items: list[str], entry: str) -> list[str]:
        deduped = [item for item in items if item != entry]
        return [entry, *deduped]

    def _compact_text(self, text: str, *, limit: int) -> str:
        compacted = " ".join(text.strip().split())
        if len(compacted) <= limit:
            return compacted
        return f"{compacted[: max(limit - 1, 1)].rstrip()}…"

    def _is_insufficient_answer(self, *, answer: str, confidence: float, evidence_count: int) -> bool:
        lowered = answer.lower()
        insufficient_markers = (
            "证据不足",
            "无法确认",
            "不能确认",
            "信息不足",
            "insufficient evidence",
            "not enough evidence",
        )
        return confidence < 0.45 or evidence_count < 2 or any(marker in lowered for marker in insufficient_markers)

    def _build_answer_quality_check(
        self,
        *,
        qa: QAResponse,
        route: str,
        scope_mode: str,
        document_ids: list[str],
    ) -> dict[str, Any]:
        evidence_count = len(qa.evidence_bundle.evidences)
        confidence = qa.confidence if qa.confidence is not None else 0.0
        insufficient = self._is_insufficient_answer(
            answer=qa.answer,
            confidence=confidence,
            evidence_count=evidence_count,
        )
        warnings: list[str] = []
        if evidence_count < 2:
            warnings.append("low_evidence_count")
        if confidence < 0.45:
            warnings.append("low_confidence")
        if route in {"document_drilldown", "chart_drilldown"} and not document_ids:
            warnings.append("drilldown_without_document_scope")
        if "无法" in qa.answer or "不能确认" in qa.answer:
            warnings.append("answer_contains_uncertainty_marker")
        return {
            "evidence_count": evidence_count,
            "confidence": round(confidence, 4),
            "route": route,
            "scope_mode": scope_mode,
            "needs_recovery": insufficient,
            "recommended_recovery": (
                "import_or_expand_evidence"
                if insufficient
                else "none"
            ),
            "warnings": warnings,
        }
