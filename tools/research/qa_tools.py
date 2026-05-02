from __future__ import annotations

import logging
from typing import Any

from domain.schemas.api import QAResponse
from domain.schemas.evidence import EvidenceBundle
from domain.schemas.research import (
    PaperCandidate,
    ResearchReport,
    ResearchTask,
    ResearchTaskAskRequest,
)
from tools.research.qa_schemas import ResearchQARouteDecision
from domain.schemas.research_context import ResearchExecutionContext
from tools.research.collection_qa_capability import ResearchCollectionQACapability
from tools.research.knowledge_access import ResearchKnowledgeAccess

logger = logging.getLogger(__name__)


class ResearchQAToolset:
    """Route-specific QA tools used by the task-level ResearchQAAgent."""

    def __init__(self, research_service: Any) -> None:
        self.research_service = research_service

    async def run(
        self,
        *,
        graph_runtime: Any,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
        qa_route_decision: ResearchQARouteDecision,
    ) -> QAResponse:
        if qa_route_decision.route == "chart_drilldown":
            return await self._run_chart_drilldown(
                graph_runtime=graph_runtime,
                task=task,
                request=request,
                papers=papers,
                document_ids=document_ids,
                execution_context=execution_context,
                qa_route_decision=qa_route_decision,
            )
        if qa_route_decision.route == "document_drilldown":
            return await self._run_document_drilldown(
                graph_runtime=graph_runtime,
                task=task,
                request=request,
                document_ids=document_ids,
                execution_context=execution_context,
                qa_route_decision=qa_route_decision,
            )
        return await self._run_collection_qa(
            graph_runtime=graph_runtime,
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
        )

    async def _run_chart_drilldown(
        self,
        *,
        graph_runtime: Any,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
        qa_route_decision: ResearchQARouteDecision,
    ) -> QAResponse:
        if qa_route_decision.visual_anchor is None and not document_ids:
            return self._blocked_chart_response(
                request=request,
                qa_route_decision=qa_route_decision,
            )
        if qa_route_decision.visual_anchor is None:
            return await self._run_document_drilldown(
                graph_runtime=graph_runtime,
                task=task,
                request=request,
                document_ids=document_ids,
                execution_context=execution_context,
                qa_route_decision=qa_route_decision,
            )

        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        visual_anchor_figure = self.research_service.chart_analysis_agent.resolve_visual_anchor_figure(
            papers=papers,
            qa_metadata={"visual_anchor": qa_route_decision.visual_anchor},
            load_cached_figure_payload=self.research_service._load_cached_figure_payload,
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
            "ResearchQAAgent running fused chart drilldown: image_path=%s chart_id=%s page_id=%s page_number=%s document_count=%s",
            str(qa_route_decision.visual_anchor.get("image_path") or ""),
            str(qa_route_decision.visual_anchor.get("chart_id") or ""),
            str(qa_route_decision.visual_anchor.get("page_id") or ""),
            qa_route_decision.visual_anchor.get("page_number"),
            len(document_ids),
        )
        fused_result = await knowledge_access.ask_fused(
            question=request.question,
            image_path=str(qa_route_decision.visual_anchor.get("image_path") or ""),
            doc_id=document_ids[0] if len(document_ids) == 1 else None,
            document_ids=document_ids,
            top_k=request.top_k,
            session_id=execution_context.session_id,
            filters=self._qa_filters(
                task=task,
                request=request,
                qa_route=qa_route_decision.route,
            ),
            metadata={
                **request.metadata,
                "qa_route": qa_route_decision.route,
                "qa_route_source": "ResearchQAAgent",
                "visual_anchor": qa_route_decision.visual_anchor,
                "visual_anchor_figure": (
                    visual_anchor_figure.model_dump(mode="json")
                    if visual_anchor_figure is not None
                    else None
                ),
                "figure_context": figure_context,
            },
            reasoning_style=request.reasoning_style,
            page_id=str(qa_route_decision.visual_anchor.get("page_id") or "") or None,
            page_number=int(qa_route_decision.visual_anchor.get("page_number") or 1),
            chart_id=str(qa_route_decision.visual_anchor.get("chart_id") or "") or None,
        )
        qa = fused_result.qa
        return qa.model_copy(
            update={
                "metadata": {
                    **qa.metadata,
                    "autonomy_mode": "task_scoped_drilldown",
                    "agent_architecture": "supervisor_to_research_qa_agent",
                    "primary_agents": ["ResearchQAAgent"],
                    "primary_tools": ["ResearchKnowledgeAccess.ask_fused"],
                    "memory_enabled": execution_context.memory_enabled,
                    "session_id": execution_context.session_id,
                    "drilldown_runtime": "fused_chart",
                }
            }
        )

    async def _run_document_drilldown(
        self,
        *,
        graph_runtime: Any,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
        qa_route_decision: ResearchQARouteDecision,
    ) -> QAResponse:
        knowledge_access = ResearchKnowledgeAccess.from_runtime(graph_runtime)
        logger.info(
            "ResearchQAAgent running document drilldown: route=%s document_count=%s has_visual_anchor=%s",
            qa_route_decision.route,
            len(document_ids),
            qa_route_decision.visual_anchor is not None,
        )
        qa = await knowledge_access.ask_document(
            question=request.question,
            document_ids=document_ids,
            top_k=request.top_k,
            filters=self._qa_filters(
                task=task,
                request=request,
                qa_route=qa_route_decision.route,
            ),
            session_id=execution_context.session_id,
            task_intent=f"research_{qa_route_decision.route}",
            metadata={
                **request.metadata,
                "qa_route": qa_route_decision.route,
                "qa_route_source": "ResearchQAAgent",
            },
            reasoning_style=request.reasoning_style,
        )
        return qa.model_copy(
            update={
                "metadata": {
                    **qa.metadata,
                    "autonomy_mode": "task_scoped_drilldown",
                    "agent_architecture": "supervisor_to_research_qa_agent",
                    "primary_agents": ["ResearchQAAgent"],
                    "primary_tools": ["ResearchKnowledgeAccess.ask_document"],
                    "memory_enabled": execution_context.memory_enabled,
                    "session_id": execution_context.session_id,
                    "drilldown_runtime": "document",
                }
            }
        )

    async def _run_collection_qa(
        self,
        *,
        graph_runtime: Any,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        report: ResearchReport | None,
        papers: list[PaperCandidate],
        document_ids: list[str],
        execution_context: ResearchExecutionContext,
    ) -> QAResponse:
        from tools.research.qa_decisions import rewrite_collection_question

        resolved_question = rewrite_collection_question(
            question=request.question,
            task=task,
            papers=papers,
            scope_mode=str((request.metadata or {}).get("qa_scope_mode") or "all_imported"),
        )
        capability = getattr(
            self.research_service,
            "research_collection_qa_capability",
            ResearchCollectionQACapability(
                llm_adapter=getattr(self.research_service, "llm_adapter", None),
            ),
        )
        return await capability.run_collection_qa(
            graph_runtime=graph_runtime,
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            resolved_question=resolved_question,
            original_question=request.question,
            primary_agents=[
                "ResearchSupervisorAgent",
                "ResearchQAAgent",
            ],
        )

    def _blocked_chart_response(
        self,
        *,
        request: ResearchTaskAskRequest,
        qa_route_decision: ResearchQARouteDecision,
    ) -> QAResponse:
        selected_titles = [
            str(item).strip()
            for item in (
                request.metadata.get("selected_paper_titles", [])
                if isinstance(request.metadata, dict)
                else []
            )
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
                "qa_route_source": "ResearchQAAgent",
                "chart_drilldown_blocked_reason": "missing_document_scope_and_visual_anchor",
            },
        )

    def _qa_filters(
        self,
        *,
        task: ResearchTask,
        request: ResearchTaskAskRequest,
        qa_route: str,
    ) -> dict[str, Any]:
        return {
            "research_task_id": task.task_id,
            "research_topic": task.topic,
            "qa_mode": "research_collection",
            "qa_route": qa_route,
            "qa_scope_mode": request.metadata.get("qa_scope_mode"),
            "selected_paper_ids": request.metadata.get("selected_paper_ids", []),
            "selected_document_ids": request.metadata.get("selected_document_ids", []),
        }
