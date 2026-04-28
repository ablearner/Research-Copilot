"""Analyze papers supervisor tool."""

from __future__ import annotations

import logging

from agents.paper_analysis_agent import PaperAnalysisAgent
from agents.research_knowledge_agent import merge_retrieval_hits
from agents.research_supervisor_agent import ResearchSupervisorDecision
from domain.schemas.research import PaperCandidate
from domain.schemas.retrieval import RetrievalHit
from services.research.supervisor_tools.base import (
    ResearchAgentToolContext,
    ResearchToolResult,
)
from services.research.supervisor_tools.mixins import _PlannerMessageTool, _WorkspacePersistenceMixin
from services.research.unified_action_adapters import (
    build_paper_analysis_input,
    build_paper_analysis_output,
    resolve_active_message,
)

logger = logging.getLogger(__name__)


class AnalyzePapersTool(_WorkspacePersistenceMixin, _PlannerMessageTool):
    name = "analyze_papers"

    def __init__(self, *, paper_analysis_agent: PaperAnalysisAgent) -> None:
        self.paper_analysis_agent = paper_analysis_agent

    async def run(self, context: ResearchAgentToolContext, decision: ResearchSupervisorDecision) -> ResearchToolResult:
        active_message = resolve_active_message(decision)
        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for selected-paper analysis",
                metadata={"reason": "missing_task"},
            )
        payload = dict(active_message.payload or {}) if active_message is not None else {}
        selected_paper_ids = [
            str(item).strip()
            for item in (payload.get("paper_ids") or context.request.selected_paper_ids)
            if str(item).strip()
        ]
        papers = self._comparison_scope_papers(
            papers=task_response.papers,
            selected_paper_ids=selected_paper_ids,
        )
        if not papers:
            return ResearchToolResult(
                status="skipped",
                observation="no papers are available for selected-paper analysis",
                metadata={"reason": "no_papers"},
            )
        analysis_input = build_paper_analysis_input(
            context=context,
            task_response=task_response,
            payload=payload,
            papers=papers,
        )
        evidence_hits = await self._collect_analysis_evidence(
            context=context,
            question=analysis_input.resolved_question(),
            papers=analysis_input.papers,
        )
        analysis = await self.paper_analysis_agent.analyze(
            question=analysis_input.resolved_question(),
            papers=analysis_input.papers,
            task_topic=analysis_input.task_topic,
            report_highlights=analysis_input.report_highlights,
            evidence_hits=evidence_hits,
        )
        context.paper_analysis_result = analysis
        self._persist_workspace_results(context, paper_analysis=analysis, analyzed_papers=analysis_input.papers)
        output = build_paper_analysis_output(
            task_id=task_response.task.task_id,
            analysis=analysis,
            analyzed_papers=analysis_input.papers,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"paper analysis completed; papers={len(analysis_input.papers)}; "
                f"focus={analysis.focus}; evidence_hits={len(evidence_hits)}"
            ),
            metadata=output.to_metadata(),
        )

    async def _collect_analysis_evidence(
        self,
        *,
        context: ResearchAgentToolContext,
        question: str,
        papers: list[PaperCandidate],
    ) -> list[RetrievalHit]:
        document_ids = list(
            dict.fromkeys(
                str(paper.metadata.get("document_id") or "").strip()
                for paper in papers
                if str(paper.metadata.get("document_id") or "").strip()
            )
        )
        if not document_ids:
            return []
        retrieval_tools = getattr(context.graph_runtime, "retrieval_tools", None)
        if retrieval_tools is None:
            return []
        execution_context = context.execution_context
        scope_filters = {
            "analysis_mode": "paper_analysis",
            "selected_paper_ids": [paper.paper_id for paper in papers],
            "selected_document_ids": document_ids,
        }
        retrieval_output = await retrieval_tools.retrieve(
            question=question,
            document_ids=document_ids,
            top_k=max(8, min(16, len(document_ids) * 4)),
            filters={
                "research_task_id": context.task.task_id if context.task is not None else None,
                "research_topic": context.task.topic if context.task is not None else "",
                **scope_filters,
            },
            session_id=getattr(execution_context, "session_id", None),
            task_id=context.task.task_id if context.task is not None else None,
            memory_hints=getattr(execution_context, "memory_hints", None) or {},
            skill_context=(
                context.graph_runtime.resolve_skill_context(
                    task_type="analyze_papers",
                    preferred_skill_name=context.request.skill_name,
                )
                if hasattr(context.graph_runtime, "resolve_skill_context")
                else None
            ),
        )
        retrieval_hits = [
            self._attach_paper_id_to_hit(hit=hit, papers=papers)
            for hit in list(retrieval_output.retrieval_result.hits or [])
        ]
        summary_hits: list[RetrievalHit] = []
        if hasattr(context.graph_runtime, "query_graph_summary"):
            summary_output = await context.graph_runtime.query_graph_summary(
                question=question,
                document_ids=document_ids,
                top_k=max(3, min(6, len(document_ids) * 2)),
                filters={
                    "research_task_id": context.task.task_id if context.task is not None else None,
                    "research_topic": context.task.topic if context.task is not None else "",
                    **scope_filters,
                },
                session_id=getattr(execution_context, "session_id", None),
                task_id=context.task.task_id if context.task is not None else None,
                memory_hints=getattr(execution_context, "memory_hints", None) or {},
                skill_context=(
                    context.graph_runtime.resolve_skill_context(
                        task_type="analyze_papers",
                        preferred_skill_name=context.request.skill_name,
                    )
                    if hasattr(context.graph_runtime, "resolve_skill_context")
                    else None
                ),
            )
            summary_hits = [
                self._attach_paper_id_to_hit(hit=hit, papers=papers)
                for hit in list(getattr(summary_output, "hits", []) or [])
            ]
        return merge_retrieval_hits(retrieval_hits, summary_hits)[:12]

    def _attach_paper_id_to_hit(
        self,
        *,
        hit: RetrievalHit,
        papers: list[PaperCandidate],
    ) -> RetrievalHit:
        document_id = str(hit.document_id or "").strip()
        matched_paper = next(
            (
                paper
                for paper in papers
                if str(paper.metadata.get("document_id") or "").strip() == document_id
            ),
            None,
        )
        if matched_paper is None:
            return hit
        metadata = dict(hit.metadata)
        metadata.setdefault("paper_id", matched_paper.paper_id)
        metadata.setdefault("title", matched_paper.title)
        return hit.model_copy(update={"metadata": metadata})
