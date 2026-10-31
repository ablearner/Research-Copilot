"""Task-level research QA execution and specialist dispatch."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Backward-compatible exports; implementation belongs to the RAG worker module.
from agents.rag_react_qa_worker import (
    RagReActQAWorker as RagReActQAWorker,
    ReActDecision as ReActDecision,
    ReActStep as ReActStep,
    ReActFinalDraft as ReActFinalDraft,
    ResearchQAAgentError as ResearchQAAgentError,
)
from domain.schemas.research import (
    DEFAULT_AGENT_REASONING_STYLE, normalize_reasoning_style,  # noqa: F401
    ResearchTask, ResearchReport, ResearchTaskAskResponse,
)
from domain.schemas.research_context import ResearchExecutionContext
from tools.research.visual_intent import VisualIntentDecision, VisualIntentRoutingTool

if TYPE_CHECKING:
    from runtime.research.agent_protocol.base import (
        ResearchAgentToolContext,
        ResearchToolResult,
    )


@dataclass
class QAExecutionResult:
    """Internal result wrapper from execute_qa() that exposes intermediate products.

    When ``persist=True`` (Service path), the caller only needs ``response``.
    When ``persist=False`` (Specialist path), the caller uses the extra fields
    to construct a ``ResearchStateDelta`` for the Runtime to persist.
    """

    response: ResearchTaskAskResponse
    updated_task: ResearchTask | None = None
    updated_report: ResearchReport | None = None
    quality_check: dict[str, Any] = field(default_factory=dict)
    persist_params: dict[str, Any] = field(default_factory=dict)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task-level ResearchQAAgent
# ---------------------------------------------------------------------------

class ResearchQAAgent:
    """Supervisor-delegated QA specialist.

    The supervisor owns task understanding and route selection. This specialist
    executes the selected QA route through the QA toolset and returns a
    unified worker result.
    """

    name = "ResearchQAAgent"
    _STREAMING_DEFAULT_CONFIDENCE: float = 0.7

    # ------------------------------------------------------------------
    # New unified entry point (SpecialistAgent protocol)
    # ------------------------------------------------------------------

    async def run_action(
        self,
        context: ResearchAgentToolContext,
        decision: Any,
    ) -> ResearchToolResult:
        from runtime.research.agent_protocol.base import MemoryOp, ResearchStateDelta, ResearchToolResult
        from runtime.research.unified_action_adapters import (
            build_collection_qa_input,
            build_collection_qa_output,
            resolve_active_message,
        )

        task_response = context.task_response
        if task_response is None:
            return ResearchToolResult(
                status="skipped",
                observation="no research task is available for collection QA",
                metadata={"reason": "missing_task"},
            )

        request = context.request
        active_message = resolve_active_message(decision)
        qa_input = build_collection_qa_input(
            context=context,
            task_id=task_response.task.task_id,
            active_message=active_message,
        )
        visual_intent = await self._decide_visual_intent_for_context(
            context=context,
            question=qa_input.question,
            explicit_image_path=qa_input.image_path,
        )
        if visual_intent.search_new_figure and not qa_input.image_path:
            return ResearchToolResult(
                status="skipped",
                observation="the user asked to locate a new visual, so chart QA should not reuse the current image anchor",
                metadata={
                    "reason": "visual_search_requires_figure_selection",
                    "visual_intent": visual_intent.model_dump(mode="json"),
                    "observation_envelope": {
                        "progress_made": False,
                        "confidence": visual_intent.confidence,
                        "suggested_next_actions": ["analyze_paper_figures"],
                        "state_delta": {
                            "target_description": visual_intent.target_description or qa_input.question,
                            "exclude_figure_ids": list(visual_intent.exclude_figure_ids),
                        },
                    },
                },
            )
        exec_result = await self.execute_qa(
            context.research_service,
            task_response.task.task_id,
            qa_input.to_research_task_ask_request(),
            graph_runtime=context.graph_runtime,
            persist=False,
        )
        qa_result = exec_result.response

        # --- P1: Build state delta from QAExecutionResult (Runtime will apply + persist) ---
        memory_ops: list[MemoryOp] = []
        current_ec = context.execution_context
        session_id = current_ec.session_id if current_ec is not None else None
        if session_id and qa_result.paper_ids:
            memory_ops.append(MemoryOp(
                op_type="set_active_papers",
                params={"session_id": session_id, "paper_ids": list(qa_result.paper_ids)},
            ))
        pp = exec_result.persist_params
        if pp.get("persist_research_update"):
            memory_ops.append(MemoryOp(
                op_type="persist_research_update",
                params=pp["persist_research_update"],
            ))
        if pp.get("promote_conclusion"):
            memory_ops.append(MemoryOp(
                op_type="promote_conclusion",
                params=pp["promote_conclusion"],
            ))
        refreshed_task_response = context.research_service.get_task(task_response.task.task_id)
        delta = ResearchStateDelta(
            task=exec_result.updated_task,
            report=exec_result.updated_report,
            qa_result=qa_result,
            task_response=refreshed_task_response,
            save_task_conversation_id=request.conversation_id,
            rebuild_execution_context=True,
            rebuild_execution_context_params={
                "graph_runtime": context.graph_runtime,
                "conversation_id": request.conversation_id,
                "task": refreshed_task_response.task if refreshed_task_response else None,
                "report": refreshed_task_response.report if refreshed_task_response else None,
                "papers": refreshed_task_response.papers if refreshed_task_response else None,
                "document_ids": list(qa_result.document_ids),
                "selected_paper_ids": list(qa_result.paper_ids),
                "skill_name": request.skill_name,
                "reasoning_style": request.reasoning_style,
                "metadata": qa_result.qa.metadata if isinstance(qa_result.qa.metadata, dict) else request.metadata,
            },
            memory_ops=memory_ops or None,
        )

        output = build_collection_qa_output(qa_result=qa_result)
        metadata = output.to_metadata()
        qa_metadata = qa_result.qa.metadata if isinstance(qa_result.qa.metadata, dict) else {}
        quality_check = qa_metadata.get("answer_quality_check")
        if (
            isinstance(quality_check, dict)
            and quality_check.get("needs_recovery")
            and not self._is_local_fallback_answer(qa_metadata)
        ):
            recovery_route = str(quality_check.get("suggested_recovery_qa_route") or "").strip()
            if recovery_route:
                metadata.update(
                    {
                        "reason": "qa_route_replan_requested",
                        "observation_envelope": {
                            "progress_made": False,
                            "confidence": qa_result.qa.confidence,
                            "suggested_next_actions": ["answer_question"],
                            "state_delta": {
                                "preferred_qa_route": recovery_route,
                                "qa_recovery_reason": quality_check.get("suggested_recovery_rationale"),
                            },
                        },
                    }
                )
                return ResearchToolResult(
                    status="skipped",
                    observation=(
                        "research QA produced an under-supported answer; supervisor replan requested "
                        f"with qa_route={recovery_route}"
                    ),
                    metadata=metadata,
                    state_delta=delta,
                )
        return ResearchToolResult(
            status="succeeded",
            observation=(
                f"answered research collection question; evidence={len(qa_result.qa.evidence_bundle.evidences)}; "
                f"confidence={qa_result.qa.confidence if qa_result.qa.confidence is not None else 'empty'}"
            ),
            metadata=metadata,
            state_delta=delta,
        )

    @staticmethod
    def _is_local_fallback_answer(metadata: dict[str, Any]) -> bool:
        """Return True when answer synthesis already fell back after provider failure.

        Such answers are low-confidence by design. Asking the supervisor to
        re-run the same QA route usually repeats the same provider timeout and
        can trap the graph in an answer_question loop.
        """

        return bool(metadata.get("fallback"))

    async def _decide_visual_intent_for_context(
        self,
        *,
        context: ResearchAgentToolContext,
        question: str,
        explicit_image_path: str | None,
    ) -> VisualIntentDecision:
        current_anchor = self._explicit_anchor_from_values(
            image_path=explicit_image_path or getattr(context.request, "chart_image_path", None),
            page_id=getattr(context.request, "page_id", None),
            page_number=getattr(context.request, "page_number", None),
            chart_id=getattr(context.request, "chart_id", None),
        )
        workspace = getattr(context, "workspace", None)
        workspace_metadata = workspace.metadata if workspace is not None and isinstance(workspace.metadata, dict) else {}
        if current_anchor is None:
            current_anchor = self._anchor_from_metadata(workspace_metadata)
        current_figure = self._figure_from_metadata(workspace_metadata)
        router = self._visual_intent_router(context.research_service)
        return await router.decide_async(
            question=question,
            current_visual_anchor=current_anchor,
            current_figure=current_figure,
            has_new_image=bool(explicit_image_path),
        )

    @staticmethod
    def _visual_intent_router(research_service: Any) -> VisualIntentRoutingTool:
        router = getattr(research_service, "visual_intent_router", None)
        if router is not None:
            return router
        llm_adapter = getattr(getattr(research_service, "chart_analysis_agent", None), "llm_adapter", None)
        if llm_adapter is None:
            llm_adapter = getattr(research_service, "llm_adapter", None)
        return VisualIntentRoutingTool(llm_adapter=llm_adapter)

    @staticmethod
    def _explicit_anchor_from_values(
        *,
        image_path: str | None,
        page_id: str | None,
        page_number: int | None,
        chart_id: str | None,
    ) -> dict[str, Any] | None:
        resolved_image_path = str(image_path or "").strip()
        if not resolved_image_path:
            return None
        anchor: dict[str, Any] = {"image_path": resolved_image_path}
        if page_id:
            anchor["page_id"] = str(page_id)
        if page_number:
            anchor["page_number"] = page_number
        if chart_id:
            anchor["chart_id"] = str(chart_id)
        return anchor

    @staticmethod
    def _anchor_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
        for key in ("latest_paper_figure_analysis", "last_visual_anchor", "visual_anchor"):
            value = metadata.get(key)
            if isinstance(value, dict) and str(value.get("image_path") or "").strip():
                return dict(value)
        figure = metadata.get("last_visual_anchor_figure")
        if isinstance(figure, dict) and str(figure.get("image_path") or "").strip():
            return dict(figure)
        return None

    @staticmethod
    def _figure_from_metadata(metadata: dict[str, Any]) -> dict[str, Any] | None:
        for key in ("last_visual_anchor_figure", "latest_paper_figure_analysis"):
            value = metadata.get(key)
            if isinstance(value, dict):
                return dict(value)
        return None

    def _restore_visual_figure_from_workspace(self, *, task: Any, report: Any | None) -> dict[str, Any] | None:
        workspaces = [
            getattr(task, "workspace", None),
            getattr(report, "workspace", None) if report is not None else None,
        ]
        for workspace in workspaces:
            metadata = getattr(workspace, "metadata", None)
            if isinstance(metadata, dict):
                figure = self._figure_from_metadata(metadata)
                if figure is not None:
                    return figure
        return None

    @staticmethod
    def _has_visual_route_signal(
        *,
        request: Any,
        visual_intent: VisualIntentDecision,
        visual_anchor: dict[str, Any] | None,
    ) -> bool:
        del visual_anchor
        if bool(getattr(request, "image_path", None)):
            return True
        if visual_intent.reuse_current_anchor or visual_intent.search_new_figure:
            return True
        signals = visual_intent.marker_signals if isinstance(visual_intent.marker_signals, dict) else {}
        return any(
            bool(signals.get(key))
            for key in ("current_visual", "new_visual_search", "negative_feedback", "visual_target")
        )

    @staticmethod
    def _fallback_route_for_non_visual_question(
        *,
        scope_mode: str,
        document_ids: list[str],
        paper_ids: list[str],
    ) -> str:
        if document_ids or len(paper_ids) == 1 or scope_mode in {"selected_documents", "selected_papers"}:
            return "document_drilldown"
        return "collection_qa"

    # ------------------------------------------------------------------
    # Core QA execution (inlined from ResearchQAExecutor)
    # ------------------------------------------------------------------

    async def execute_qa(
        self, research_service: Any, task_id: str, request: Any, *,
        graph_runtime: Any, persist: bool = True,
    ) -> QAExecutionResult:
        """Compatibility entrypoint; application service owns loading and commit."""
        return await research_service.qa_service.execute_qa(
            task_id, request, graph_runtime=graph_runtime, persist=persist, executor=self,
        )

    async def execute_loaded_qa(
        self, dependencies: Any, task: Any, report: Any, papers: list[Any], request: Any, *,
        graph_runtime: Any,
        execution_context: ResearchExecutionContext,
    ) -> QAExecutionResult:
        """Execute QA against loaded inputs, returning proposed workflow changes."""
        from domain.schemas.research import ResearchTaskAskResponse
        from tools.research.qa_decisions import build_answer_quality_check, select_recovery_qa_route
        from tools.research.qa_schemas import ResearchQARouteDecision
        from tools.research.qa_tools import ResearchQAToolset, scope_qa_execution_context

        svc = dependencies
        tools = ResearchQAToolset(
            chart_analysis_agent=svc.chart_analysis_agent,
            load_cached_figure_payload=svc._load_cached_figure_payload,
            collection_qa_capability=svc.research_collection_qa_capability,
        )
        task_id = task.task_id
        scope = svc.paper_selector_service.resolve_qa_scope(
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
        explicit_visual_anchor = svc._extract_visual_anchor(request=request, metadata=request.metadata)
        restored_visual_anchor = svc._restore_visual_anchor_from_workspace(task=task, report=report)
        restored_visual_figure = self._restore_visual_figure_from_workspace(task=task, report=report)
        visual_intent = await self._visual_intent_router(svc).decide_async(
            question=request.question,
            current_visual_anchor=explicit_visual_anchor or restored_visual_anchor,
            current_figure=restored_visual_figure,
            has_new_image=bool(request.image_path),
        )
        reusable_visual_anchor = explicit_visual_anchor or (
            restored_visual_anchor if visual_intent.reuse_current_anchor else None
        )
        visual_route_allowed = self._has_visual_route_signal(
            request=request,
            visual_intent=visual_intent,
            visual_anchor=reusable_visual_anchor,
        )
        ignored_preferred_qa_route: str | None = None

        routing_authority = str(request.metadata.get("routing_authority") or "").strip()
        preferred_qa_route = str(request.metadata.get("preferred_qa_route") or "").strip()
        if preferred_qa_route == "chart_drilldown" and not visual_route_allowed:
            ignored_preferred_qa_route = preferred_qa_route
            preferred_qa_route = ""
        user_intent = None
        if routing_authority != "supervisor_llm":
            user_intent = await svc.user_intent_resolver.resolve_async(
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
                has_visual_anchor=reusable_visual_anchor is not None,
                has_document_input=False,
            )
            resolved_intent_paper_ids = [
                paper_id
                for paper_id in user_intent.resolved_paper_ids
                if paper_id in {paper.paper_id for paper in papers}
            ]
            if resolved_intent_paper_ids and not scope.paper_ids:
                scope = svc.paper_selector_service.resolve_qa_scope(
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
        if (
            routing_authority == "supervisor_llm"
            and preferred_qa_route not in allowed_supervisor_routes
            and ignored_preferred_qa_route is None
        ):
            raise ValueError(
                "Supervisor-authorized research QA must include preferred_qa_route="
                "collection_qa, document_drilldown, or chart_drilldown."
            )
        if preferred_qa_route in allowed_supervisor_routes:
            qa_route_decision = ResearchQARouteDecision(
                route=preferred_qa_route,
                confidence=0.99,
                rationale="Supervisor selected the QA route explicitly.",
                visual_anchor=reusable_visual_anchor,
            )
        else:
            qa_route_decision = await svc._select_qa_route(
                question=request.question,
                scope_mode=scope.scope_mode,
                paper_ids=scope.paper_ids,
                document_ids=document_ids,
                request=request,
                metadata=request.metadata,
            )
            if visual_intent.search_new_figure and qa_route_decision.visual_anchor is not None:
                qa_route_decision = ResearchQARouteDecision(
                    route=qa_route_decision.route,
                    confidence=qa_route_decision.confidence,
                    rationale=f"{qa_route_decision.rationale} Visual intent requested a new figure search.",
                    visual_anchor=None,
                    recovery_count=qa_route_decision.recovery_count,
                )

        if visual_intent.search_new_figure and qa_route_decision.route != "chart_drilldown":
            qa_route_decision = ResearchQARouteDecision(
                route="chart_drilldown",
                confidence=max(qa_route_decision.confidence, visual_intent.confidence, 0.85),
                rationale=(
                    f"{qa_route_decision.rationale} Visual intent requested locating a new figure, "
                    "so chart drilldown is required."
                ),
                visual_anchor=None,
                recovery_count=qa_route_decision.recovery_count,
            )

        if (
            qa_route_decision.route == "chart_drilldown"
            and qa_route_decision.visual_anchor is None
            and not visual_route_allowed
            and ignored_preferred_qa_route is not None
        ):
            fallback_route = self._fallback_route_for_non_visual_question(
                scope_mode=scope.scope_mode,
                document_ids=document_ids,
                paper_ids=scope.paper_ids,
            )
            qa_route_decision = ResearchQARouteDecision(
                route=fallback_route,  # type: ignore[arg-type]
                confidence=min(qa_route_decision.confidence, 0.74),
                rationale=(
                    f"{qa_route_decision.rationale} Chart drilldown was ignored because the question "
                    "does not contain a visual reference and no visual anchor or image is available."
                ),
                visual_anchor=None,
                recovery_count=qa_route_decision.recovery_count,
            )

        logger.info(
            "ResearchQAAgent route selected: route=%s confidence=%.2f has_visual_anchor=%s",
            qa_route_decision.route,
            qa_route_decision.confidence,
            qa_route_decision.visual_anchor is not None,
        )
        if (
            qa_route_decision.route == "chart_drilldown"
            and qa_route_decision.visual_anchor is None
            and visual_route_allowed
        ):
            inferred_visual_anchor = await svc._infer_or_discover_visual_anchor(
                task_id=task.task_id,
                papers=scoped_papers,
                document_ids=document_ids,
                question=request.question,
                graph_runtime=graph_runtime,
                exclude_figure_ids=visual_intent.exclude_figure_ids,
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
        if (
            qa_route_decision.route == "chart_drilldown"
            and qa_route_decision.visual_anchor is None
            and visual_intent.reuse_current_anchor
        ):
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
                    "visual_intent": visual_intent.model_dump(mode="json"),
                    "routing_authority": routing_authority or None,
                    "preferred_qa_route": preferred_qa_route or None,
                    "ignored_preferred_qa_route": ignored_preferred_qa_route,
                }
            }
        )
        execution_context = scope_qa_execution_context(
            execution_context,
            papers=scoped_papers,
            document_ids=document_ids,
            selected_paper_ids=scope.paper_ids,
            context_manager=svc.research_context_manager,
            paper_reading_tool=svc.paper_reading_tool,
        )
        qa = await tools.run(
            graph_runtime=graph_runtime,
            task=task,
            request=scoped_request,
            report=report,
            papers=scoped_papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa_route_decision=qa_route_decision,
        )
        quality_check = build_answer_quality_check(
            qa=qa,
            route=qa_route_decision.route,
            scope_mode=scope.scope_mode,
            document_ids=document_ids,
        )
        if routing_authority == "supervisor_llm":
            recovery_candidate = select_recovery_qa_route(
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
            qa, qa_route_decision, quality_check = await self._recover_qa_route(
                tools=tools,
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
                    "visual_intent": visual_intent.model_dump(mode="json"),
                    "ignored_preferred_qa_route": ignored_preferred_qa_route,
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
                    "qa_executor": "ResearchQAAgent",
                }
            }
        )
        qa_metadata = qa.metadata if isinstance(qa.metadata, dict) else {}
        visual_anchor_figure = svc.chart_analysis_agent.resolve_visual_anchor_figure(
            papers=scoped_papers,
            qa_metadata=qa_metadata,
            load_cached_figure_payload=svc._load_cached_figure_payload,
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

        updated_task, updated_report = svc._apply_qa_follow_up(
            task=task,
            report=report,
            request=scoped_request,
            qa=qa,
            papers=papers,
            document_ids=document_ids,
            scope=scope,
        )
        persist_research_update_params = {
            "session_id": execution_context.session_id,
            "conversation_id": request.conversation_id,
            "graph_runtime": graph_runtime,
            "task": updated_task,
            "report": updated_report,
            "papers": scoped_papers,
            "document_ids": document_ids,
            "selected_paper_ids": scope.paper_ids,
            "task_intent": f"research_{qa_route_decision.route}:{scope.scope_mode}",
            "question": request.question,
            "answer": qa.answer,
            "retrieval_summary": (
                f"documents={len(document_ids)}, evidences={len(qa.evidence_bundle.evidences)}, "
                f"confidence={qa.confidence if qa.confidence is not None else 'empty'}"
            ),
            "metadata_update": {
                "last_skill_name": request.skill_name,
                "reasoning_style": request.reasoning_style or "cot",
                "evidence_count": len(qa.evidence_bundle.evidences),
                "qa_scope_mode": scope.scope_mode,
                "selected_paper_ids": scope.paper_ids[:8],
                "answer_quality_check": quality_check,
            },
        }
        promote_conclusion_params = None
        if not quality_check["needs_recovery"] and execution_context.session_id:
            promote_conclusion_params = {
                "session_id": execution_context.session_id,
                "conclusion": svc._compact_text(qa.answer, limit=700),
                "topic": task.topic,
                "keywords": [qa_route_decision.route, scope.scope_mode],
                "related_paper_ids": scope.paper_ids[:8],
                "metadata": {
                    "question": request.question,
                    "confidence": qa.confidence,
                    "evidence_count": len(qa.evidence_bundle.evidences),
                },
            }
        response = ResearchTaskAskResponse(
            task_id=task_id,
            paper_ids=scope.paper_ids,
            document_ids=document_ids,
            scope_mode=scope.scope_mode,
            qa=qa,
            report=updated_report,
            todo_items=updated_task.todo_items,
            warnings=scope.warnings,
        )
        return QAExecutionResult(
            response=response,
            updated_task=updated_task,
            updated_report=updated_report,
            quality_check=quality_check,
            persist_params={
                "persist_research_update": persist_research_update_params,
                "promote_conclusion": promote_conclusion_params,
                "conversation_id": request.conversation_id,
            },
        )

    async def _recover_qa_route(
        self,
        *,
        tools: Any,
        graph_runtime: Any,
        task: Any,
        request: Any,
        report: Any | None,
        papers: list[Any],
        document_ids: list[str],
        execution_context: Any,
        qa: Any,
        qa_route_decision: Any,
        quality_check: dict[str, Any],
        scope: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        from tools.research.qa_decisions import (
            build_answer_quality_check,
            select_recovery_qa_route,
        )

        rerouted = select_recovery_qa_route(
            request=request,
            scope=scope,
            document_ids=document_ids,
            qa=qa,
            qa_route_decision=qa_route_decision,
            quality_check=quality_check,
        )
        if rerouted is None:
            return qa, qa_route_decision, quality_check
        recovered_qa = await tools.run(
            graph_runtime=graph_runtime,
            task=task,
            request=request,
            report=report,
            papers=papers,
            document_ids=document_ids,
            execution_context=execution_context,
            qa_route_decision=rerouted,
        )
        recovered_quality = build_answer_quality_check(
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
