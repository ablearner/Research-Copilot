from __future__ import annotations

import logging
from typing import Any

from domain.schemas.research import ResearchTaskAskRequest, ResearchTaskAskResponse
from services.research.qa.schemas import ResearchQARouteDecision
from services.research.qa.tools import ResearchQAToolset

logger = logging.getLogger(__name__)


class ResearchQAExecutor:
    """Task-level QA executor used by ResearchQAAgent and legacy adapters."""

    def __init__(self, research_service: Any) -> None:
        self.research_service = research_service
        self.tools = ResearchQAToolset(research_service)

    async def execute(
        self,
        task_id: str,
        request: ResearchTaskAskRequest,
        *,
        graph_runtime: Any,
    ) -> ResearchTaskAskResponse:
        task = self.research_service.report_service.load_task(task_id)
        if task is None:
            raise KeyError(task_id)
        report = self.research_service.report_service.load_report(task.task_id, task.report_id)
        papers = self.research_service.report_service.load_papers(task.task_id)
        scope = self.research_service.paper_selector_service.resolve_qa_scope(
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
            raise ValueError(
                f"Research task has no imported documents or persisted research artifacts available for QA: {task_id}"
            )

        routing_authority = str(request.metadata.get("routing_authority") or "").strip()
        preferred_qa_route = str(request.metadata.get("preferred_qa_route") or "").strip()
        user_intent = None
        if routing_authority != "supervisor_llm":
            user_intent = await self.research_service.user_intent_resolver.resolve_async(
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
                    for item in (
                        (request.metadata.get("context") or {}).get("active_paper_ids", [])
                        if isinstance(request.metadata.get("context"), dict)
                        else []
                    )
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
                scope = self.research_service.paper_selector_service.resolve_qa_scope(
                    task=task,
                    papers=papers,
                    requested_paper_ids=resolved_intent_paper_ids,
                    requested_document_ids=request.document_ids,
                )
                document_ids = list(scope.document_ids)
                scoped_papers = list(scope.papers or papers)
            if user_intent.needs_clarification:
                raise ValueError(user_intent.clarification_question or "当前问题指向不明确，请补充具体论文或图表。")

        allowed_supervisor_routes = {"collection_qa", "document_drilldown", "chart_drilldown"}
        if routing_authority == "supervisor_llm" and preferred_qa_route not in allowed_supervisor_routes:
            raise ValueError(
                "Supervisor-authorized research QA must include preferred_qa_route="
                "collection_qa, document_drilldown, or chart_drilldown."
            )
        if preferred_qa_route in allowed_supervisor_routes:
            qa_route_decision = ResearchQARouteDecision(
                route=preferred_qa_route,
                confidence=0.99,
                rationale="Supervisor selected the QA route explicitly.",
                visual_anchor=self.research_service._extract_visual_anchor(request=request, metadata=request.metadata),
            )
        else:
            qa_route_decision = await self.research_service._select_qa_route(
                question=request.question,
                scope_mode=scope.scope_mode,
                paper_ids=scope.paper_ids,
                document_ids=document_ids,
                request=request,
                metadata=request.metadata,
            )

        logger.info(
            "ResearchQAAgent route selected: route=%s confidence=%.2f has_visual_anchor=%s",
            qa_route_decision.route,
            qa_route_decision.confidence,
            qa_route_decision.visual_anchor is not None,
        )
        if qa_route_decision.route == "chart_drilldown" and qa_route_decision.visual_anchor is None:
            inferred_visual_anchor = await self.research_service._infer_or_discover_visual_anchor(
                task_id=task.task_id,
                papers=scoped_papers,
                document_ids=document_ids,
                question=request.question,
                graph_runtime=graph_runtime,
            )
            if inferred_visual_anchor is not None:
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
            restored_visual_anchor = self.research_service._restore_visual_anchor_from_workspace(task=task, report=report)
            if restored_visual_anchor is not None:
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
        execution_context = self.research_service.build_execution_context(
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
        qa = await self.tools.run(
            graph_runtime=graph_runtime,
            task=task,
            request=scoped_request,
            report=report,
            papers=scoped_papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa_route_decision=qa_route_decision,
        )
        quality_check = self.research_service._build_answer_quality_check(
            qa=qa,
            route=qa_route_decision.route,
            scope_mode=scope.scope_mode,
            document_ids=document_ids,
        )
        if routing_authority == "supervisor_llm":
            recovery_candidate = self.research_service._select_recovery_qa_route(
                request=scoped_request,
                scope=scope,
                document_ids=document_ids,
                qa=qa,
                qa_route_decision=qa_route_decision,
                quality_check=quality_check,
            )
            if recovery_candidate is not None:
                quality_check = {
                    **quality_check,
                    "needs_recovery": True,
                    "suggested_recovery_qa_route": recovery_candidate.route,
                    "suggested_recovery_rationale": recovery_candidate.rationale,
                    "suggested_recovery_confidence": recovery_candidate.confidence,
                }
        else:
            qa, qa_route_decision, quality_check = await self._recover_legacy_route(
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
                    "answer_quality_check": quality_check,
                    "qa_executor": "ResearchQAExecutor",
                }
            }
        )
        qa_metadata = qa.metadata if isinstance(qa.metadata, dict) else {}
        visual_anchor_figure = self.research_service.chart_analysis_agent.resolve_visual_anchor_figure(
            papers=scoped_papers,
            qa_metadata=qa_metadata,
            load_cached_figure_payload=self.research_service._load_cached_figure_payload,
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

        updated_task, updated_report = self.research_service._apply_qa_follow_up(
            task=task,
            request=scoped_request,
            qa=qa,
            papers=papers,
            document_ids=document_ids,
            scope=scope,
        )
        self.research_service.save_task_state(updated_task, conversation_id=request.conversation_id)
        self.research_service.report_service.save_report(updated_report)
        self.research_service.memory_gateway.persist_research_update(
            session_id=execution_context.session_id,
            conversation_id=request.conversation_id,
            graph_runtime=graph_runtime,
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
            self.research_service.memory_gateway.promote_conclusion_to_long_term(
                execution_context.session_id,
                conclusion=self.research_service._compact_text(qa.answer, limit=700),
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

    async def _recover_legacy_route(
        self,
        *,
        graph_runtime: Any,
        task: Any,
        request: ResearchTaskAskRequest,
        report: Any | None,
        papers: list[Any],
        document_ids: list[str],
        execution_context: Any,
        qa: Any,
        qa_route_decision: ResearchQARouteDecision,
        quality_check: dict[str, Any],
        scope: Any,
    ) -> tuple[Any, ResearchQARouteDecision, dict[str, Any]]:
        rerouted = self.research_service._select_recovery_qa_route(
            request=request,
            scope=scope,
            document_ids=document_ids,
            qa=qa,
            qa_route_decision=qa_route_decision,
            quality_check=quality_check,
        )
        if rerouted is None:
            return qa, qa_route_decision, quality_check
        recovered_qa = await self.tools.run(
            graph_runtime=graph_runtime,
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa_route_decision=rerouted,
        )
        recovered_quality = self.research_service._build_answer_quality_check(
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
