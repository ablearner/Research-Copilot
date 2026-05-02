from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from tools.research import PaperAnalyzer

from domain.schemas.research import PaperCandidate
from domain.schemas.retrieval import RetrievalHit
from domain.schemas.research_functions import AnalyzePapersFunctionOutput

if TYPE_CHECKING:
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )

logger = logging.getLogger(__name__)

_ANALYSIS_EVIDENCE_TIMEOUT_SECONDS = 15.0
_UNQUERYABLE_INDEX_STATUSES = {"timeout", "failed", "skipped"}


def _has_queryable_index(paper: PaperCandidate) -> bool:
    metadata = paper.metadata if isinstance(paper.metadata, dict) else {}
    document_id = str(metadata.get("document_id") or "").strip()
    if not document_id:
        return False
    index_status = str(metadata.get("index_status") or "").strip().lower()
    if index_status in _UNQUERYABLE_INDEX_STATUSES:
        return False
    if metadata.get("indexed") is False:
        return False
    return True


class PaperAnalysisAgent:
    """Top-level worker agent for selected-paper analysis tasks."""

    name = "PaperAnalysisAgent"

    def __init__(
        self,
        *,
        paper_analysis_skill: PaperAnalyzer,
    ) -> None:
        self.paper_analysis_skill = paper_analysis_skill

    def _dedupe_ids(self, values: list[str]) -> list[str]:
        return list(dict.fromkeys(values))

    def _dedupe_text(self, values: list[str], *, limit: int) -> list[str]:
        deduped = [v.strip() for v in values if v and v.strip()]
        return list(dict.fromkeys(deduped))[:limit]

    # ------------------------------------------------------------------
    # New unified entry point (SpecialistAgent protocol)
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
    ) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import MemoryOp, ResearchStateDelta, ResearchToolResult
        from runtime.research.agent_protocol.mixins import comparison_scope_papers, persist_workspace_results
        from runtime.research.unified_action_adapters import (
            build_paper_analysis_input,
            build_paper_analysis_output,
            resolve_active_message,
        )

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
        papers = comparison_scope_papers(
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
        analysis = await self.analyze(
            question=analysis_input.resolved_question(),
            papers=analysis_input.papers,
            task_topic=analysis_input.task_topic,
            report_highlights=analysis_input.report_highlights,
            evidence_hits=evidence_hits,
            supervisor_instruction=context.supervisor_instruction,
        )
        ws_result = persist_workspace_results(
            context,
            paper_analysis=analysis,
            analyzed_papers=analysis_input.papers,
            persist=False,
        )
        output = build_paper_analysis_output(
            task_id=task_response.task.task_id,
            analysis=analysis,
            analyzed_papers=analysis_input.papers,
        )
        memory_ops: list[MemoryOp] = []
        if ws_result is not None and ws_result.memory_save_context_params is not None:
            memory_ops.append(MemoryOp(
                op_type="save_context",
                params=ws_result.memory_save_context_params,
            ))
        delta = ResearchStateDelta(
            task=ws_result.updated_task if ws_result else None,
            report=ws_result.updated_report if ws_result else None,
            save_task_conversation_id=context.request.conversation_id,
            save_task_event_type=ws_result.save_event_type if ws_result else None,
            save_task_event_payload=ws_result.save_event_payload if ws_result else None,
            task_response=ws_result.updated_task_response if ws_result else None,
            paper_analysis_result=analysis,
            memory_ops=memory_ops or None,
        )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"paper analysis completed; papers={len(analysis_input.papers)}; "
                f"focus={analysis.focus}; evidence_hits={len(evidence_hits)}"
            ),
            metadata=output.to_metadata(),
            state_delta=delta,
        )

    async def _collect_analysis_evidence(
        self,
        *,
        context: ResearchAgentToolContext,
        question: str,
        papers: list[PaperCandidate],
    ) -> list[RetrievalHit]:
        from domain.schemas.retrieval import merge_retrieval_hits
        from tools.research.knowledge_access import ResearchKnowledgeAccess

        queryable_papers = [paper for paper in papers if _has_queryable_index(paper)]
        document_ids = list(
            dict.fromkeys(
                str(paper.metadata.get("document_id") or "").strip()
                for paper in queryable_papers
                if str(paper.metadata.get("document_id") or "").strip()
            )
        )
        if not document_ids:
            return []
        knowledge_access = context.knowledge_access or ResearchKnowledgeAccess.from_runtime(context.graph_runtime)
        execution_context = context.execution_context
        scope_filters = {
            "analysis_mode": "paper_analysis",
            "selected_paper_ids": [paper.paper_id for paper in queryable_papers],
            "selected_document_ids": document_ids,
        }
        try:
            retrieval_output = await asyncio.wait_for(
                knowledge_access.retrieve(
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
                ),
                timeout=_ANALYSIS_EVIDENCE_TIMEOUT_SECONDS,
            )
        except (RuntimeError, asyncio.TimeoutError) as exc:
            logger.warning("Paper analysis retrieval skipped: %s", exc)
            retrieval_hits = []
        else:
            retrieval_hits = [
                self._attach_paper_id_to_hit(hit=hit, papers=queryable_papers)
                for hit in list(retrieval_output.retrieval_result.hits or [])
            ]
        try:
            summary_output = await asyncio.wait_for(
                knowledge_access.query_graph_summary(
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
                ),
                timeout=_ANALYSIS_EVIDENCE_TIMEOUT_SECONDS,
            )
        except (RuntimeError, asyncio.TimeoutError) as exc:
            logger.warning("Paper analysis graph summary skipped: %s", exc)
            summary_hits = []
        else:
            summary_hits = [
                self._attach_paper_id_to_hit(hit=hit, papers=queryable_papers)
                for hit in list(getattr(summary_output, "hits", []) or [])
            ]
        return merge_retrieval_hits(retrieval_hits, summary_hits)[:12]

    def _attach_paper_id_to_hit(self, *, hit: RetrievalHit, papers: list[PaperCandidate]) -> RetrievalHit:
        document_id = str(hit.document_id or "").strip()
        matched_paper = next(
            (paper for paper in papers if str(paper.metadata.get("document_id") or "").strip() == document_id),
            None,
        )
        if matched_paper is None:
            return hit
        metadata = dict(hit.metadata)
        metadata.setdefault("paper_id", matched_paper.paper_id)
        metadata.setdefault("title", matched_paper.title)
        return hit.model_copy(update={"metadata": metadata})

    # ------------------------------------------------------------------
    # Legacy unified runtime entry point (will be removed in Step 6)
    # ------------------------------------------------------------------

    async def analyze(
        self,
        *,
        question: str,
        papers: list[PaperCandidate],
        task_topic: str = "",
        report_highlights: list[str] | None = None,
        evidence_hits: list[RetrievalHit] | None = None,
        supervisor_instruction: str | None = None,
    ) -> AnalyzePapersFunctionOutput:
        return await self.paper_analysis_skill.analyze_async(
            question=question,
            papers=papers,
            task_topic=task_topic,
            report_highlights=report_highlights or [],
            evidence_hits=evidence_hits or [],
            supervisor_instruction=supervisor_instruction,
        )
