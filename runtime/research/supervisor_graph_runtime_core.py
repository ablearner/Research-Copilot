from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from agents.chart_analysis_agent import ChartAnalysisAgent
from agents.general_answer_agent import GeneralAnswerAgent
from agents.literature_scout_agent import LiteratureScoutAgent
from agents.paper_analysis_agent import PaperAnalysisAgent
from agents.preference_memory_agent import PreferenceMemoryAgent
from agents.research_document_agent import ResearchDocumentAgent
from agents.research_knowledge_agent import ResearchKnowledgeAgent
from agents.research_qa_agent import ResearchQAAgent
from agents.research_supervisor_agent import (
    ResearchSupervisorAgent,
    ResearchSupervisorDecision,
    ResearchSupervisorState,
)
from agents.research_writer_agent import ResearchWriterAgent
from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research import (
    PaperCandidate,
    ResearchAdvancedStrategy,
    ResearchAgentRunRequest,
    ResearchAgentRunResponse,
    ResearchAgentTraceStep,
    ResearchMessage,
    ResearchTaskResponse,
    ResearchWorkspaceState,
)
from domain.schemas.sub_manager import SubManagerState, TaskStep
from domain.schemas.unified_runtime import (
    UNIFIED_ACTION_OUTPUT_METADATA_KEY,
    UnifiedAgentDescriptor,
    UnifiedAgentResult,
    UnifiedAgentTask,
)
from runtime.research.unified_runtime import (
    serialize_unified_delegation_plan,
)
from typing import TYPE_CHECKING
from domain.research_workspace import build_workspace_from_task, build_workspace_state
from tools.research import PaperAnalyzer, PaperReader, ResearchIntentResolver, PaperCurator
from tools.research.user_intent import ResearchUserIntentResult
from tooling.executor import ToolExecutor
from tooling.registry import ToolRegistry
from context.compressor import ContextCompressor
from core.skill_registry import SkillRegistry
from core.skill_matcher import SkillMatcher
from core.utils import normalize_topic_text as _normalize_topic_text_impl
from runtime.research.context_builder import ResearchAgentContextBuilder
from runtime.research.response_formatter import ResearchResponseFormatter
from runtime.research.result_aggregator import ResearchAgentResultAggregator
from tools.research.capability_registry import ResearchCapabilityRegistry
from tools.research.skill_resolver import ResearchSkillResolver

from runtime.research.agent_protocol import (
    ResearchAgentGraphState,
    ResearchAgentToolContext,
    ResearchStateDelta,
    ResearchToolResult,
    _message,
    _update_runtime_progress,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from services.research.literature_research_service import LiteratureResearchService


# ---------------------------------------------------------------------------
# Intent classification heuristics (extracted to intent_classifier.py)
# ---------------------------------------------------------------------------

from runtime.research.intent_classifier import (
    _route_mode_hint_for_request,
    _should_inherit_snapshot_scope,
    resolve_intent_flags,
    should_force_finalize as _should_force_finalize_impl,
)


class ResearchRuntimeBase:
    """Shared specialist runtime capabilities for high-level research orchestration."""

    def __init__(
        self,
        *,
        research_service: LiteratureResearchService,
        manager_agent: ResearchSupervisorAgent | None = None,
        max_steps: int = 8,
    ) -> None:
        self.research_service = research_service
        paper_reading_skill = getattr(research_service, "paper_reading_skill", None) or PaperReader()
        llm_adapter = getattr(getattr(research_service, "paper_search_service", None), "llm_adapter", None)
        if llm_adapter is None:
            llm_adapter = getattr(paper_reading_skill, "llm_adapter", None)
        self.manager_agent = manager_agent or ResearchSupervisorAgent(llm_adapter=llm_adapter)
        if getattr(self.manager_agent, "llm_adapter", None) is None:
            self.manager_agent.llm_adapter = llm_adapter
        self.user_intent_resolver = ResearchIntentResolver(llm_adapter=llm_adapter)
        self.max_steps = max_steps
        self.literature_scout_agent = LiteratureScoutAgent(
            research_service.paper_search_service,
            llm_adapter=llm_adapter,
        )
        self.research_knowledge_agent = ResearchKnowledgeAgent(
            llm_adapter=llm_adapter,
        )
        self.research_document_agent = ResearchDocumentAgent()
        self.research_qa_agent = ResearchQAAgent()
        self.research_writer_agent = ResearchWriterAgent(
            research_service.paper_search_service,
            llm_adapter=llm_adapter,
        )
        self.paper_analysis_agent = PaperAnalysisAgent(
            paper_analysis_skill=PaperAnalyzer(
                paper_reading_skill=paper_reading_skill,
                llm_adapter=llm_adapter,
            )
        )
        self.chart_analysis_agent = ChartAnalysisAgent(llm_adapter=llm_adapter)
        self.general_answer_agent = GeneralAnswerAgent(llm_adapter=llm_adapter)
        self.preference_memory_agent = getattr(research_service, "preference_memory_agent", None) or PreferenceMemoryAgent(
            memory_manager=research_service.memory_manager,
            memory_gateway=getattr(research_service, "memory_gateway", None),
            paper_search_service=research_service.paper_search_service,
            storage_root=research_service.report_service.storage_root,
            llm_adapter=llm_adapter,
        )
        self.literature_scout_agent.research_writer_agent = self.research_writer_agent
        self.paper_curation_skill = PaperCurator(research_service.paper_search_service)
        self.literature_scout_agent.curation_skill = self.paper_curation_skill
        self._context_compressor = ContextCompressor(
            llm_adapter=llm_adapter,
            target_budget_ratio=0.75,
        )
        self.unified_agent_delegates = self._build_unified_agent_delegates()
        self.tool_registry = ToolRegistry()
        self.action_tool_executor = ToolExecutor(self.tool_registry)
        self.skill_registry = SkillRegistry()
        self.skill_matcher = SkillMatcher(self.skill_registry)
        self.skill_registry.scan()
        self.skill_resolver = ResearchSkillResolver(
            registry=self.skill_registry,
            matcher=self.skill_matcher,
        )
        self.context_builder = ResearchAgentContextBuilder(
            runtime=self,
            skill_resolver=self.skill_resolver,
        )
        self.capability_registry = ResearchCapabilityRegistry(runtime=self)
        self.response_formatter = ResearchResponseFormatter(
            manager_display_name=self._manager_display_name(),
        )
        self.result_aggregator = ResearchAgentResultAggregator(runtime=self)

    @property
    def action_tool_registry(self) -> ToolRegistry:
        """Backward-compatible alias — returns the unified tool registry."""
        return self.tool_registry

    def _manager_display_name(self) -> str:
        return "ResearchSupervisorAgent"

    def _manager_trace_agent_name(self) -> str:
        return self._manager_display_name()

    def _hydrate_request_from_conversation(
        self,
        *,
        request: ResearchAgentRunRequest,
    ) -> tuple[ResearchAgentRunRequest, ResearchTaskResponse | None]:
        if not request.conversation_id:
            return request, None
        try:
            conversation_response = self.research_service.get_conversation(request.conversation_id)
        except KeyError:
            return request, None

        conversation = conversation_response.conversation
        snapshot = conversation.snapshot

        restored_task_response: ResearchTaskResponse | None = None
        if request.task_id:
            try:
                restored_task_response = self.research_service.get_task(request.task_id)
            except KeyError:
                restored_task_response = None
        elif conversation.task_id:
            try:
                restored_task_response = self.research_service.get_task(conversation.task_id)
            except KeyError:
                restored_task_response = snapshot.task_result
        elif snapshot.task_result is not None:
            restored_task_response = snapshot.task_result

        inherit_scope = _should_inherit_snapshot_scope(request=request, snapshot=snapshot)
        selected_paper_ids = list(request.selected_paper_ids)
        if not selected_paper_ids and inherit_scope:
            selected_paper_ids = list(
                snapshot.selected_paper_ids
                or snapshot.active_paper_ids
                or []
            )
        selected_document_ids = list(request.selected_document_ids)
        if not selected_document_ids and selected_paper_ids and restored_task_response is not None:
            imported_document_ids = {
                str(item).strip()
                for item in restored_task_response.task.imported_document_ids
                if str(item).strip()
            }
            papers_by_id = {paper.paper_id: paper for paper in restored_task_response.papers}
            for paper_id in selected_paper_ids:
                paper = papers_by_id.get(paper_id)
                if paper is None:
                    continue
                document_id = str(paper.metadata.get("document_id") or "").strip()
                if not document_id:
                    continue
                if imported_document_ids and document_id not in imported_document_ids:
                    continue
                if document_id not in selected_document_ids:
                    selected_document_ids.append(document_id)
        if (
            not selected_document_ids
            and inherit_scope
            and snapshot.ask_result is not None
        ):
            selected_document_ids = [
                str(item).strip()
                for item in snapshot.ask_result.document_ids
                if str(item).strip()
            ]

        metadata = dict(request.metadata or {})
        metadata_context = (
            dict(metadata.get("context") or {})
            if isinstance(metadata.get("context"), dict)
            else {}
        )
        active_paper_ids = [
            str(item).strip()
            for item in metadata_context.get("active_paper_ids", [])
            if str(item).strip()
        ]
        if not active_paper_ids and inherit_scope:
            active_paper_ids = list(snapshot.active_paper_ids or snapshot.selected_paper_ids or [])
        if active_paper_ids:
            metadata_context["active_paper_ids"] = active_paper_ids
            metadata["context"] = metadata_context
        elif "context" in metadata and isinstance(metadata.get("context"), dict):
            metadata_context.pop("active_paper_ids", None)
            metadata["context"] = metadata_context

        heuristic_intent = self.user_intent_resolver.resolve(
            message=request.message,
            has_task=restored_task_response is not None,
            candidate_paper_count=len(restored_task_response.papers) if restored_task_response else 0,
            candidate_papers=None,
            active_paper_ids=active_paper_ids,
            selected_paper_ids=selected_paper_ids,
        )
        metadata_context["route_mode"] = _route_mode_hint_for_request(
            request=request,
            snapshot=snapshot,
            inherit_scope=inherit_scope,
            intent_result=heuristic_intent,
        )
        metadata_context["active_thread_id"] = snapshot.active_thread_id
        metadata["context"] = metadata_context
        metadata["user_intent"] = heuristic_intent.model_dump(mode="json")

        hydrated_request = request.model_copy(
            update={
                "task_id": request.task_id or (
                    restored_task_response.task.task_id if restored_task_response is not None else None
                ),
                "selected_paper_ids": selected_paper_ids,
                "selected_document_ids": selected_document_ids,
                "metadata": metadata,
            }
        )
        return hydrated_request, restored_task_response

    async def _build_tool_context(self, *, request: ResearchAgentRunRequest, graph_runtime: Any) -> ResearchAgentToolContext:
        self._inject_skill_matcher_backends(graph_runtime)
        return await self.context_builder.build(request=request, graph_runtime=graph_runtime)

    def _inject_skill_matcher_backends(self, graph_runtime: Any) -> None:
        """Lazy-inject embedding_adapter and reranker into the skill matcher.

        These are only available from graph_runtime (RagRuntime), which is not
        known at __init__ time.  Injection happens once on first request.
        """
        if self.skill_matcher.embedding_adapter is not None:
            return  # already injected
        embedding_adapter = getattr(
            getattr(graph_runtime, "embedding_index_service", None),
            "embedding_adapter",
            None,
        )
        if embedding_adapter is not None:
            self.skill_matcher.embedding_adapter = embedding_adapter
        reranker = getattr(
            getattr(
                getattr(graph_runtime, "retrieval_tools", None),
                "retriever",
                None,
            ),
            "reranker",
            None,
        )
        if reranker is not None:
            self.skill_matcher.reranker = reranker
            # Warmup: trigger tokenizer/model weight loading to avoid cold-start
            try:
                reranker._predict([("warmup", "warmup")])
            except Exception:
                pass

    def _build_unified_agent_delegates(self) -> dict[str, Any]:
        return {
            "ResearchSupervisorAgent": self.manager_agent,
            "LiteratureScoutAgent": self.literature_scout_agent,
            "ResearchKnowledgeAgent": self.research_knowledge_agent,
            "ResearchDocumentAgent": self.research_document_agent,
            "ResearchQAAgent": self.research_qa_agent,
            "ResearchWriterAgent": self.research_writer_agent,
            "PaperAnalysisAgent": self.paper_analysis_agent,
            "PreferenceMemoryAgent": self.preference_memory_agent,
            "ChartAnalysisAgent": self.chart_analysis_agent,
            "GeneralAnswerAgent": self.general_answer_agent,
        }

    def _build_unified_execution_handlers(self) -> dict[str, Any]:
        return {}

    def _with_standardized_observation(
        self,
        *,
        action_name: str,
        context: ResearchAgentToolContext,
        status: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        from runtime.research.agent_protocol.base import _observation_envelope
        next_actions = list((context.workspace.next_actions if context.workspace is not None else [])[:3])
        observation = dict(metadata.get("observation_envelope") or {})
        progress_made = status == "succeeded" and metadata.get("reason") not in {"route_mismatch"}
        confidence = metadata.get("confidence")
        if action_name == "answer_question" and context.qa_result is not None:
            confidence = context.qa_result.qa.confidence
            next_actions = list((context.qa_result.report.workspace.next_actions if context.qa_result.report is not None else next_actions)[:3])
        elif action_name == "general_answer":
            confidence = metadata.get("confidence")
        elif action_name == "search_literature":
            confidence = 0.82 if status == "succeeded" else 0.35
        elif action_name == "write_review":
            confidence = 0.78 if status == "succeeded" else 0.4
        elif action_name == "analyze_papers":
            confidence = 0.8 if status == "succeeded" else 0.4
        elif action_name == "recommend_from_preferences":
            confidence = 0.83 if status == "succeeded" else 0.35
        elif action_name == "compress_context":
            confidence = 0.74 if status == "succeeded" else 0.3
        missing_inputs = list(observation.get("missing_inputs") or [])
        reason = str(metadata.get("reason") or "").strip()
        if reason == "missing_task":
            missing_inputs.append("task")
        elif reason == "missing_document_file_path":
            missing_inputs.append("document_file_path")
        elif reason == "missing_chart_image_path":
            missing_inputs.append("chart_image_path")
        elif reason == "no_target_papers":
            missing_inputs.append("paper_scope")
        elif reason == "no_papers":
            missing_inputs.append("papers")
        elif reason == "missing_execution_context":
            missing_inputs.append("execution_context")
        suggested = list(observation.get("suggested_next_actions") or [])
        if not suggested:
            if status == "skipped" and reason in {"missing_task", "no_target_papers", "no_papers"}:
                suggested = ["clarify_request", "search_literature"]
            elif action_name == "answer_question":
                suggested = ["analyze_papers", "compress_context"]
            elif action_name == "search_literature":
                suggested = ["write_review", "import_papers", "answer_question"]
            elif action_name == "recommend_from_preferences":
                suggested = ["search_literature", "answer_question"]
            elif action_name == "general_answer" and metadata.get("reason") == "route_mismatch":
                suggested = [str(metadata.get("suggested_action") or "answer_question")]
            elif next_actions:
                suggested = next_actions
        state_delta = dict(observation.get("state_delta") or {})
        if action_name == "search_literature" and context.task is not None:
            state_delta.setdefault("task_id", context.task.task_id)
            state_delta.setdefault("paper_count", len(context.papers))
        if action_name == "answer_question" and context.qa_result is not None:
            state_delta.setdefault("qa_task_id", context.qa_result.task_id)
            state_delta.setdefault("paper_ids", list(context.qa_result.paper_ids))
        if action_name == "recommend_from_preferences" and context.preference_recommendation_result is not None:
            state_delta.setdefault(
                "recommended_paper_ids",
                [item.paper_id for item in context.preference_recommendation_result.recommendations],
            )
        if action_name == "general_answer":
            state_delta.setdefault("ignore_research_context", bool(metadata.get("ignore_research_context")))
        metadata["observation_envelope"] = _observation_envelope(
            progress_made=progress_made,
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            missing_inputs=list(dict.fromkeys(item for item in missing_inputs if item)),
            suggested_next_actions=list(dict.fromkeys(str(item).strip() for item in suggested if str(item).strip()))[:4],
            state_delta=state_delta,
        )
        return metadata


    _TASK_TYPE_AGENTS = {"ResearchKnowledgeAgent", "ChartAnalysisAgent"}

    async def _execute_agent_run_action(
        self,
        *,
        action_name: str,
        context: ResearchAgentToolContext,
        decision: ResearchSupervisorDecision,
        worker_agent: str,
    ) -> ResearchToolResult | None:
        agent = self.unified_agent_delegates.get(worker_agent)
        if agent is None or not hasattr(agent, "run_action"):
            return None
        kwargs: dict[str, Any] = {}
        if worker_agent in self._TASK_TYPE_AGENTS:
            kwargs["task_type"] = action_name
        try:
            result = await agent.run_action(context, decision, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s.run_action() failed", worker_agent, exc_info=True)
            return ResearchToolResult(
                status="failed",
                observation=str(exc) or f"{worker_agent} run_action failed",
                metadata={
                    "reason": "tool_execution_failed",
                    "specialist_error_type": exc.__class__.__name__,
                    "execution_engine": "unified_agent_registry",
                    "execution_adapter": f"agent_run_action:{agent.name}",
                },
            )
        metadata = dict(result.metadata)
        metadata.setdefault("execution_engine", "unified_agent_registry")
        metadata.setdefault("execution_adapter", f"agent_run_action:{agent.name}")
        enriched_metadata = self._with_standardized_observation(
            action_name=action_name,
            context=context,
            status=result.status,
            metadata=metadata,
        )
        return ResearchToolResult(
            status=result.status,
            observation=result.observation,
            metadata=enriched_metadata,
            state_delta=result.state_delta,
        )

    async def _decide_next_action(self, state: ResearchAgentGraphState) -> ResearchSupervisorDecision:
        context = state["context"]
        session_ctx = self._resolve_session_context(context)
        user_intent = await self.user_intent_resolver.resolve_async(
            message=context.request.message,
            has_task=context.task is not None,
            candidate_paper_count=len(context.papers),
            candidate_papers=self._candidate_paper_scope_for_manager(context.papers),
            active_paper_ids=self._active_paper_ids_for_manager(context),
            selected_paper_ids=list(context.request.selected_paper_ids),
            has_visual_anchor=bool(context.request.chart_image_path or context.request.chart_id),
            has_document_input=bool(context.request.document_file_path),
            session_topic=session_ctx["previous_topic"] or None,
        )

        # --- Fast path: skip Supervisor LLM for trivial messages ---
        fast_decision = self._try_fast_route(
            context=context,
            user_intent=user_intent,
            session_ctx=session_ctx,
            state=state,
        )
        if fast_decision is not None:
            return self._inject_skill_metadata_into_decision(fast_decision, context=context)

        supervisor_state = self._state_from_context(
            context,
            trace=state.get("trace", []),
            user_intent=user_intent,
            session_context=session_ctx,
            execution_plan=state.get("execution_plan", []),
        )
        decision = await self.manager_agent.decide_next_action_async(
            supervisor_state,
            pending_messages=state.get("pending_agent_messages", []),
            agent_messages=state.get("agent_messages", []),
            agent_results=state.get("agent_results", []),
            completed_task_ids=state.get("completed_agent_task_ids", []),
            failed_task_ids=state.get("failed_agent_task_ids", []),
            replanned_failure_task_ids=state.get("replanned_failure_task_ids", []),
            planner_runs=int(state.get("planner_runs", 0) or 0),
            replan_count=int(state.get("replan_count", 0) or 0),
            context_slice=self._context_slice(context),
            clarification_request=state.get("clarification_request"),
            active_plan_id=state.get("active_plan_id"),
        )
        return self._inject_skill_metadata_into_decision(decision, context=context)

    def _try_fast_route(
        self,
        *,
        context: ResearchAgentToolContext,
        user_intent: Any,
        session_ctx: dict[str, Any],
        state: ResearchAgentGraphState,
    ) -> ResearchSupervisorDecision | None:
        """Return a pre-built decision for trivial messages, or None to fall through to LLM."""
        route_mode = session_ctx.get("route_mode", "")
        intent_name = str(getattr(user_intent, "intent", "") or "").strip()
        has_task = context.task is not None
        is_first_step = int(state.get("current_step_index", 0) or 0) == 0

        # Greeting / general chat without active research task → general_answer
        if (
            route_mode == "general_chat"
            and intent_name in ("general_answer", "general_follow_up")
            and not has_task
            and is_first_step
            and not context.request.document_file_path
            and not context.request.chart_image_path
        ):
            logger.info("Fast-route: general_chat → general_answer (skipping Supervisor LLM)")
            return ResearchSupervisorDecision(
                action_name="general_answer",
                thought="Simple greeting or general chat detected — routing directly without LLM planning.",
                rationale="fast_route:general_chat",
                phase="act",
                estimated_gain=0.3,
                estimated_cost=0.1,
                metadata={
                    "worker_agent": "GeneralAnswerAgent",
                    "fast_route": True,
                    "route_mode": route_mode,
                    "intent": intent_name,
                },
            )
        return None

    def _inject_skill_metadata_into_decision(
        self,
        decision: ResearchSupervisorDecision,
        *,
        context: ResearchAgentToolContext,
    ) -> ResearchSupervisorDecision:
        selection = context.skill_selection
        if selection is None or not selection.active_skill_names:
            return decision
        active_skill_names = list(selection.active_skill_names)
        skill_payload = {
            "active_skill_names": active_skill_names,
            "skill_context": context.skill_context,
        }
        decision.metadata.setdefault("active_skill_names", active_skill_names)
        decision.metadata.setdefault("skill_context", context.skill_context)
        if active_skill_names:
            decision.metadata.setdefault(
                "skill_name",
                (context.request.skill_name or "").strip() or active_skill_names[0],
            )
        active_message = self._active_message(decision)
        if active_message is None:
            return decision
        preferred_skill_name = (context.request.skill_name or "").strip() or active_skill_names[0]
        message_metadata = {
            **active_message.metadata,
            **skill_payload,
            "skill_name": active_message.metadata.get("skill_name") or preferred_skill_name,
        }
        enriched_message = active_message.model_copy(update={"metadata": message_metadata})
        decision.metadata["active_message"] = enriched_message
        state_update = decision.metadata.get("state_update")
        if isinstance(state_update, dict):
            state_update["pending_agent_messages"] = [
                enriched_message
                if getattr(message, "task_id", None) == enriched_message.task_id
                else message
                for message in state_update.get("pending_agent_messages", [])
            ]
            state_update["agent_messages"] = [
                enriched_message
                if getattr(message, "task_id", None) == enriched_message.task_id
                else message
                for message in state_update.get("agent_messages", [])
            ]
        return decision

    def _route_decision(self, decision: ResearchSupervisorDecision) -> str:
        return "finalize" if decision.action_name in {"finalize", "clarify_request"} else decision.action_name

    def _progress_signature(self, state: ResearchAgentGraphState) -> str:
        context = state["context"]
        workspace = context.workspace
        workspace_metadata = workspace.metadata if workspace is not None else {}
        task = context.task
        report = context.report
        return "|".join(
            [
                str(task.task_id if task else ""),
                str(len(context.papers)),
                str(report.report_id if report else ""),
                str(len(task.imported_document_ids) if task else 0),
                str(len(task.todo_items) if task else 0),
                str(context.import_result.imported_count if context.import_result else 0),
                str(context.qa_result.qa.answer[:80] if context.qa_result else ""),
                str(context.paper_analysis_result is not None or bool(workspace_metadata.get("latest_paper_analysis"))),
                str(
                    context.preference_recommendation_result is not None
                    or bool(workspace_metadata.get("latest_preference_recommendations"))
                ),
                str(context.compressed_context_summary is not None or bool(workspace_metadata.get("context_compression"))),
                str(context.parsed_document is not None),
                str(context.chart_result is not None),
                str(len(state.get("agent_results", []))),
                str(len(state.get("trace", []))),
            ]
        )

    def _should_force_finalize(self, state: ResearchAgentGraphState) -> bool:
        context = state["context"]
        request = context.request
        latest_result = state.get("agent_results", [])[-1] if state.get("agent_results") else None
        latest_observation: dict[str, Any] = {}
        if latest_result is not None:
            observation = latest_result.payload.get("observation_envelope")
            if isinstance(observation, dict):
                latest_observation = observation
        latest_next_actions = {
            str(item).strip()
            for item in latest_observation.get("suggested_next_actions", [])
            if str(item).strip()
        }
        return _should_force_finalize_impl(
            exhausted=bool(state.get("exhausted")),
            stagnant_count=int(state.get("stagnant_decision_count", 0) or 0),
            repeated_count=int(state.get("repeated_action_count", 0) or 0),
            mode=request.mode,
            has_qa_result=context.qa_result is not None,
            latest_task_type=latest_result.task_type if latest_result is not None else None,
            latest_status=latest_result.status if latest_result is not None else None,
            latest_next_actions=latest_next_actions,
            workflow_constraint=str((request.metadata or {}).get("workflow_constraint") or "").strip(),
            has_preference_result=context.preference_recommendation_result is not None,
            advanced_action=request.advanced_action,
            has_paper_analysis=(
                context.paper_analysis_result is not None
                or self._workspace_has(context, "latest_paper_analysis")
            ),
            new_topic_detected=bool(state.get("new_topic_detected")),
            has_task_response=context.task_response is not None,
            has_report=context.report is not None,
            auto_import=request.auto_import,
            has_message=bool(request.message.strip()),
            import_attempted=context.import_attempted or context.import_result is not None,
            has_import_result=context.import_result is not None,
        )

    def _workspace_has(self, context: ResearchAgentToolContext, key: str) -> bool:
        workspace = context.workspace
        return bool(workspace is not None and workspace.metadata.get(key))

    def _forced_finalize_decision(self, state: ResearchAgentGraphState) -> ResearchSupervisorDecision:
        stagnant = int(state.get("stagnant_decision_count", 0) or 0)
        repeated = int(state.get("repeated_action_count", 0) or 0)
        reason = "Research loop reached a natural stopping point."
        if stagnant >= 2:
            reason = "Research loop stopped because workspace progress stalled across consecutive decisions."
        elif repeated >= 2:
            reason = "Research loop stopped because the manager repeated the same action without enough new progress."
        elif state.get("exhausted"):
            reason = f"Research loop stopped after reaching max_steps={self.max_steps}."
        return ResearchSupervisorDecision(
            action_name="finalize",
            thought="Stop the loop and return the current workspace instead of continuing low-value work.",
            rationale=reason,
            phase="commit",
            estimated_gain=0.0,
            estimated_cost=0.0,
            stop_reason=reason,
            metadata={
                "decision_source": "runtime_guardrail",
                "worker_agent": self._manager_trace_agent_name(),
                "loop_guardrail": {
                    "stagnant_decision_count": stagnant,
                    "repeated_action_count": repeated,
                    "progress_signature": state.get("progress_signature", ""),
                },
            },
        )

    def _on_decision(
        self,
        state: ResearchAgentGraphState,
        step_index: int,
        decision: ResearchSupervisorDecision,
    ) -> ResearchAgentGraphState:
        signature = self._progress_signature(state)
        previous_signature = str(state.get("progress_signature") or "")
        stagnant_count = int(state.get("stagnant_decision_count", 0) or 0)
        if previous_signature and signature == previous_signature:
            stagnant_count += 1
        else:
            stagnant_count = 0
        trace = state.get("trace", [])
        last_action = trace[-1].action_name if trace else None
        repeated_count = int(state.get("repeated_action_count", 0) or 0)
        if decision.action_name != "finalize" and decision.action_name == last_action:
            repeated_count += 1
        else:
            repeated_count = 0
        logger.info(
            "Research manager loop guard | step=%s | action=%s | stagnant=%s | repeated=%s | signature=%s",
            step_index,
            decision.action_name,
            stagnant_count,
            repeated_count,
            signature,
        )
        update = decision.metadata.get("state_update") if isinstance(decision.metadata, dict) else None
        if isinstance(update, dict):
            update = {
                **update,
                "progress_signature": signature,
                "stagnant_decision_count": stagnant_count,
                "repeated_action_count": repeated_count,
            }
            self._rebuild_execution_context(
                context=state["context"],
                agent_messages=update.get("agent_messages", state.get("agent_messages", [])),
                agent_results=update.get("agent_results", state.get("agent_results", [])),
                pending_messages=update.get("pending_agent_messages", state.get("pending_agent_messages", [])),
            )
            self._persist_execution_context(state["context"])
            return update  # type: ignore[return-value]
        return {
            "progress_signature": signature,
            "stagnant_decision_count": stagnant_count,
            "repeated_action_count": repeated_count,
        }

    async def search_literature_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "search_literature")

    async def write_review_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "write_review")

    async def import_papers_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "import_papers")

    async def sync_to_zotero_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "sync_to_zotero")

    async def answer_question_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "answer_question")

    async def general_answer_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "general_answer")

    async def recommend_from_preferences_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "recommend_from_preferences")

    async def analyze_papers_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "analyze_papers")

    async def compress_context_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "compress_context")

    async def understand_document_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "understand_document")

    async def understand_chart_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "supervisor_understand_chart")

    async def analyze_paper_figures_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        return await self._run_action_node(state, "analyze_paper_figures")

    # ------------------------------------------------------------------
    # Unified state-delta application (P1 refactor)
    # ------------------------------------------------------------------

    def _apply_state_delta(
        self,
        *,
        context: ResearchAgentToolContext,
        delta: ResearchStateDelta,
    ) -> None:
        """Apply context mutations from a specialist's state delta.

        This is the SINGLE entry point for specialist-produced state changes.
        Specialists MUST NOT mutate context or call persist methods directly.
        """
        if delta.task_response is not None:
            context.task_response = delta.task_response
        if delta.qa_result is not None:
            context.qa_result = delta.qa_result
        if delta.paper_analysis_result is not None:
            context.paper_analysis_result = delta.paper_analysis_result
        if delta.preference_recommendation_result is not None:
            context.preference_recommendation_result = delta.preference_recommendation_result
        if delta.import_result is not None:
            context.import_result = delta.import_result
        if delta.compressed_context_summary is not None:
            context.compressed_context_summary = delta.compressed_context_summary
        if delta.rebuild_execution_context and delta.rebuild_execution_context_params is not None:
            context.execution_context = context.research_service.build_execution_context(
                **delta.rebuild_execution_context_params,
            )

    def _persist_state_delta(
        self,
        *,
        context: ResearchAgentToolContext,
        delta: ResearchStateDelta,
    ) -> None:
        """Persist state mutations from a specialist's state delta.

        This is the SINGLE entry point for specialist-produced persistence.
        """
        svc = context.research_service
        task_id = delta.task.task_id if delta.task is not None else None

        if delta.papers is not None and task_id is not None:
            svc.report_service.save_papers(task_id, delta.papers)
        if delta.report is not None:
            svc.report_service.save_report(delta.report)
        if delta.task is not None:
            svc.save_task_state(
                delta.task,
                conversation_id=delta.save_task_conversation_id,
                event_type=delta.save_task_event_type,
                payload=delta.save_task_event_payload,
            )
        if delta.record_task_turn and delta.task_response is not None:
            conversation_id = delta.save_task_conversation_id or context.request.conversation_id
            if conversation_id:
                svc.record_task_turn(conversation_id, response=delta.task_response)

        for mem_op in (delta.memory_ops or []):
            self._execute_memory_op(context, mem_op)

    def _execute_memory_op(self, context: ResearchAgentToolContext, mem_op: Any) -> None:
        """Execute a single deferred memory operation."""
        gateway = context.research_service.memory_gateway
        op_type = mem_op.op_type
        params = dict(mem_op.params)
        try:
            if op_type == "set_active_papers":
                gateway.set_active_papers(params["session_id"], params["paper_ids"])
            elif op_type == "persist_research_update":
                gateway.persist_research_update(**params)
            elif op_type == "promote_conclusion":
                gateway.promote_conclusion_to_long_term(**params)
            elif op_type == "update_paper_knowledge":
                gateway.update_paper_knowledge(params["record"])
            elif op_type == "record_turn":
                gateway.record_turn(**params)
            elif op_type == "record_user_recommendations":
                gateway.record_user_recommendations(**params)
            elif op_type == "save_context":
                gateway.save_context(params["session_id"], params["research_context"])
            elif op_type == "record_import_turn":
                context.research_service.record_import_turn(**params)
            else:
                logger.warning("Unknown memory op type: %s", op_type)
        except Exception:
            logger.warning("Memory op %s failed", op_type, exc_info=True)

    async def _run_action_node(
        self,
        state: ResearchAgentGraphState,
        action_name: Literal[
            "search_literature",
            "write_review",
            "import_papers",
            "sync_to_zotero",
            "answer_question",
            "general_answer",
            "recommend_from_preferences",
            "analyze_papers",
            "compress_context",
            "understand_document",
            "supervisor_understand_chart",
            "analyze_paper_figures",
        ],
    ) -> ResearchAgentGraphState:
        decision = state.get("current_decision")
        if decision is None or decision.action_name != action_name:
            return {}
        _update_runtime_progress(
            state["context"],
            stage=action_name,
            node=f"{action_name}_node",
            status="running",
            summary=f"Executing {action_name}.",
            extra={"step_index": int(state.get("current_step_index", 0) or 0)},
        )
        trace = list(state.get("trace", []))
        step_index = state.get("current_step_index", 0)
        worker_agent = self._worker_agent_name(decision)
        active_message = self._active_message(decision)
        context = state["context"]
        context.supervisor_instruction = (
            active_message.instruction.strip() if active_message is not None else None
        ) or None
        result = await self._execute_agent_run_action(
            action_name=action_name,
            context=context,
            decision=decision,
            worker_agent=worker_agent,
        )
        if result is None:
            result = ResearchToolResult(
                status="failed",
                observation=f"no agent found for worker={worker_agent} action={action_name}",
                metadata={"reason": "missing_agent", "execution_engine": "unified_agent_registry"},
            )

        # --- P1: Unified state-delta application ---
        if result.state_delta is not None:
            self._apply_state_delta(context=state["context"], delta=result.state_delta)
            self._persist_state_delta(context=state["context"], delta=result.state_delta)

        status = result.status if result.status in {"succeeded", "failed", "skipped"} else "succeeded"
        trace.append(
            ResearchAgentTraceStep(
                step_index=step_index,
                agent=worker_agent,
                thought=decision.thought,
                action_name=action_name,
                phase=decision.phase,  # type: ignore[arg-type]
                action_input=decision.action_input,
                status=status,  # type: ignore[arg-type]
                observation=result.observation,
                rationale=decision.rationale,
                estimated_gain=decision.estimated_gain,
                estimated_cost=decision.estimated_cost,
                stop_signal=status == "failed",
                workspace_summary=self._workspace_summary(state["context"]),
                metadata=self._trace_metadata(
                    dict(result.metadata),
                    active_message=active_message,
                ),
            )
        )
        update = {
            "trace": trace,
            "failed": bool(state.get("failed")) or result.status == "failed",
            **self._message_result_update(
                state,
                active_message=active_message,
                worker_agent=worker_agent,
                status=status,
                payload=dict(result.metadata),
            ),
        }
        self._rebuild_execution_context(
            context=state["context"],
            agent_messages=state.get("agent_messages", []),
            agent_results=update.get("agent_results", state.get("agent_results", [])),
            pending_messages=update.get("pending_agent_messages", state.get("pending_agent_messages", [])),
        )
        return update

    def route_after_action(self, state: ResearchAgentGraphState) -> str:
        del state
        return "manager"

    async def finalize_node(self, state: ResearchAgentGraphState) -> ResearchAgentGraphState:
        decision = state.get("current_decision")
        if decision is None:
            return {}
        _update_runtime_progress(
            state["context"],
            stage="finalize",
            node="finalize_node",
            status="running",
            summary="Finalizing supervisor response.",
            extra={"step_index": int(state.get("current_step_index", 0) or 0)},
        )
        trace = list(state.get("trace", []))
        worker_agent = self._worker_agent_name(decision)
        active_message = self._active_message(decision)
        trace.append(
            ResearchAgentTraceStep(
                step_index=state.get("current_step_index", 0),
                agent=worker_agent,
                thought=decision.thought,
                action_name="finalize",
                phase=decision.phase,  # type: ignore[arg-type]
                action_input=decision.action_input,
                status="succeeded",
                observation="manager requested clarification and stopped the loop" if decision.action_name == "clarify_request" else "manager stopped the loop",
                rationale=decision.rationale,
                estimated_gain=decision.estimated_gain,
                estimated_cost=decision.estimated_cost,
                stop_signal=True,
                workspace_summary=self._workspace_summary(state["context"]),
                metadata=self._trace_metadata(
                    {
                        "stop_reason": decision.stop_reason,
                        **(
                            {"clarification_request": decision.stop_reason}
                            if decision.action_name == "clarify_request" and decision.stop_reason
                            else {}
                        ),
                    },
                    active_message=active_message,
                ),
            )
        )
        update = {
            "trace": trace,
            **self._message_result_update(
                state,
                active_message=active_message,
                worker_agent=worker_agent,
                status="succeeded",
                payload={
                    "stop_reason": decision.stop_reason,
                    **(
                        {"clarification_request": decision.stop_reason}
                        if decision.action_name == "clarify_request" and decision.stop_reason
                        else {}
                    ),
                },
            ),
        }
        self._rebuild_execution_context(
            context=state["context"],
            agent_messages=state.get("agent_messages", []),
            agent_results=update.get("agent_results", state.get("agent_results", [])),
            pending_messages=update.get("pending_agent_messages", state.get("pending_agent_messages", [])),
        )
        self._persist_execution_context(state["context"])
        return update

    def _context_slice(self, context: ResearchAgentToolContext):
        execution_context = context.execution_context
        if execution_context is None:
            return None
        return execution_context.context_slices.get("manager") or execution_context.context_slices.get("default")

    def _rebuild_execution_context(
        self,
        *,
        context: ResearchAgentToolContext,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        pending_messages: list[AgentMessage],
    ) -> None:
        """Rebuild in-memory execution context (task plan + slices). No I/O."""
        execution_context = context.execution_context
        if execution_context is None or execution_context.research_context is None:
            return
        research_context = execution_context.research_context.model_copy(deep=True)
        pending_ids = {message.task_id for message in pending_messages}
        result_by_task_id = {result.task_id: result for result in agent_results}
        task_plan: list[TaskStep] = []
        for message in agent_messages:
            result = result_by_task_id.get(message.task_id)
            status = "planned"
            if message.task_id in pending_ids:
                status = "queued"
            if result is not None:
                if result.status == "failed" or (result.evaluation is not None and not result.evaluation.passed):
                    status = "failed"
                elif result.status == "skipped":
                    status = "skipped"
                elif result.status == "succeeded":
                    status = "succeeded"
            task_plan.append(
                TaskStep(
                    task_id=message.task_id,
                    assigned_to=message.agent_to,
                    instruction=message.instruction,
                    task_type=message.task_type,
                    depends_on=list(message.depends_on),
                    context_slice=self._serialize_context_slice(message.context_slice),
                    expected_output_schema=dict(message.expected_output_schema),
                    priority=self._task_priority_from_message(message.priority),
                    retry_count=message.retry_count,
                    status=status,  # type: ignore[arg-type]
                    metadata=dict(message.metadata),
                )
            )

        research_context.current_task_plan = task_plan
        research_context.sub_manager_states = self._build_sub_manager_states(task_plan)
        execution_context.research_context = research_context
        execution_context.context_slices = self.research_service.build_context_slices(
            research_context,
            selected_paper_ids=context.request.selected_paper_ids,
        )

    def _persist_execution_context(self, context: ResearchAgentToolContext) -> None:
        """Persist the current execution context to storage."""
        execution_context = context.execution_context
        if execution_context is None or execution_context.research_context is None:
            return
        if execution_context.session_id:
            self.research_service.memory_gateway.save_context(
                execution_context.session_id,
                execution_context.research_context,
            )

    def _active_message(self, decision: ResearchSupervisorDecision) -> AgentMessage | None:
        if not isinstance(decision.metadata, dict):
            return None
        active_message = decision.metadata.get("active_message")
        if isinstance(active_message, AgentMessage):
            return active_message
        if isinstance(active_message, dict):
            return AgentMessage.model_validate(active_message)
        return None

    def _worker_agent_name(self, decision: ResearchSupervisorDecision) -> str:
        if isinstance(decision.metadata, dict):
            worker_agent = decision.metadata.get("worker_agent")
            if isinstance(worker_agent, str) and worker_agent.strip():
                return worker_agent
        return self._manager_trace_agent_name()

    def _unified_agent_descriptor(
        self,
        context: ResearchAgentToolContext,
        *,
        agent_name: str,
    ) -> UnifiedAgentDescriptor | None:
        registry = context.unified_agent_registry
        if registry is None:
            return None
        executor = registry.get(agent_name)
        if executor is None:
            return None
        return executor.descriptor

    def _preferred_skill_name_for_message(
        self,
        context: ResearchAgentToolContext,
        *,
        active_message: AgentMessage,
        worker_agent: str,
    ) -> str | None:
        for key in ("preferred_skill_name", "skill_name"):
            raw_value = active_message.metadata.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()
        request_skill_name = (context.request.skill_name or "").strip()
        if request_skill_name:
            return request_skill_name
        descriptor = self._unified_agent_descriptor(context, agent_name=worker_agent)
        if descriptor is not None and descriptor.capability_binding.profile_name:
            return descriptor.capability_binding.profile_name
        return None

    def _available_tool_names_for_agent(
        self,
        context: ResearchAgentToolContext,
        *,
        agent_name: str,
    ) -> list[str]:
        descriptor = self._unified_agent_descriptor(context, agent_name=agent_name)
        if descriptor is None:
            return []
        return list(descriptor.available_tool_names)

    def _trace_metadata(
        self,
        payload: dict[str, Any],
        *,
        active_message: AgentMessage | None,
    ) -> dict[str, Any]:
        metadata = dict(payload)
        if active_message is not None:
            metadata.setdefault("planner_task_id", active_message.task_id)
            metadata.setdefault("task_type", active_message.task_type)
            metadata.setdefault("agent_to", active_message.agent_to)
        return metadata

    def _message_result_update(
        self,
        state: ResearchAgentGraphState,
        *,
        active_message: AgentMessage | None,
        worker_agent: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> ResearchAgentGraphState:
        if active_message is None:
            return {}
        context = state["context"]
        result_payload = dict(payload or {})
        pending = [
            message
            for message in state.get("pending_agent_messages", [])
            if message.task_id != active_message.task_id
        ]
        if status == "failed":
            pending = []
        descriptor = self._unified_agent_descriptor(context, agent_name=worker_agent)
        unified_task = UnifiedAgentTask.from_agent_message(
            active_message,
            preferred_skill_name=self._preferred_skill_name_for_message(
                context,
                active_message=active_message,
                worker_agent=worker_agent,
            ),
            available_tool_names=self._available_tool_names_for_agent(
                context,
                agent_name=worker_agent,
            ),
        )
        action_output = UnifiedAgentResult.extract_action_output(payload=result_payload)
        result_metadata = {"plan_id": active_message.metadata.get("plan_id"), **result_payload}
        if action_output is not None:
            result_metadata.setdefault(
                UNIFIED_ACTION_OUTPUT_METADATA_KEY,
                dict(action_output),
            )
        if unified_task.preferred_skill_name:
            result_metadata.setdefault("preferred_skill_name", unified_task.preferred_skill_name)
        if unified_task.available_tool_names:
            result_metadata.setdefault("available_tool_names", list(unified_task.available_tool_names))
        if descriptor is not None:
            result_metadata.setdefault("execution_mode", descriptor.execution_mode)
            result_metadata.setdefault("agent_descriptor_name", descriptor.name)
        unified_result = UnifiedAgentResult(
            task_id=active_message.task_id,
            agent_name=worker_agent,
            task_type=active_message.task_type,
            status=status,  # type: ignore[arg-type]
            instruction=active_message.instruction,
            payload=result_payload,
            context_slice=active_message.context_slice,
            priority=active_message.priority,
            expected_output_schema=active_message.expected_output_schema,
            depends_on=list(active_message.depends_on),
            retry_count=active_message.retry_count,
            evaluation=result_payload.get("evaluation") if isinstance(result_payload, dict) else None,
            action_output=action_output,
            metadata=result_metadata,
        )
        result_message = unified_result.to_agent_result_message(reply_to=active_message.agent_from)
        completed = list(state.get("completed_agent_task_ids", []))
        failed = list(state.get("failed_agent_task_ids", []))
        if status in {"succeeded", "skipped"}:
            completed.append(active_message.task_id)
        if status == "failed":
            failed.append(active_message.task_id)
        return {
            "pending_agent_messages": pending,
            "agent_results": [*state.get("agent_results", []), result_message],
            "completed_agent_task_ids": self._dedupe_ids(completed),
            "failed_agent_task_ids": self._dedupe_ids(failed),
        }

    def _dedupe_ids(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            deduped.append(value)
            seen.add(value)
        return deduped

    def _serialize_context_slice(self, context_slice: Any) -> dict[str, Any]:
        if hasattr(context_slice, "model_dump"):
            return context_slice.model_dump(mode="json")
        if isinstance(context_slice, dict):
            return dict(context_slice)
        return {}

    def _task_priority_from_message(self, priority: str) -> str:
        if priority in {"high", "critical"}:
            return "high"
        if priority == "low":
            return "low"
        return "normal"

    def _build_sub_manager_states(
        self,
        task_plan: list[TaskStep],
    ) -> dict[str, SubManagerState]:
        states = {
            "research": SubManagerState(name="research"),
            "writing": SubManagerState(name="writing"),
        }
        assignments = {
            "LiteratureScoutAgent": "research",
            "ChartAnalysisAgent": "research",
            "ResearchKnowledgeAgent": "research",
            "PreferenceMemoryAgent": "research",
            "ResearchWriterAgent": "writing",
            "PaperAnalysisAgent": "writing",
        }
        for task in task_plan:
            key = assignments.get(task.assigned_to)
            if key is None:
                continue
            state = states[key]
            if task.status in {"planned", "queued", "running"}:
                state.active_task_ids = self._dedupe_ids([*state.active_task_ids, task.task_id])
            if task.status == "succeeded":
                state.completed_task_ids = self._dedupe_ids([*state.completed_task_ids, task.task_id])
            if task.status == "failed":
                state.status = "failed"
            elif task.status in {"planned", "queued", "running"} and state.status != "failed":
                state.status = "running"
            elif (
                task.status == "succeeded"
                and not state.active_task_ids
                and state.status not in {"failed", "running"}
            ):
                state.status = "completed"
            state.last_task_plan_id = str(task.metadata.get("plan_id") or state.last_task_plan_id or "")
        for state in states.values():
            if state.status == "idle" and state.completed_task_ids:
                state.status = "completed"
            if not state.last_task_plan_id:
                state.last_task_plan_id = None
        return states

    def _serialize_task_plan(
        self,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
    ) -> list[dict[str, Any]]:
        result_by_task_id = {result.task_id: result for result in agent_results}
        serialized: list[dict[str, Any]] = []
        for message in agent_messages:
            result = result_by_task_id.get(message.task_id)
            serialized_status = "planned"
            if result is not None:
                serialized_status = (
                    "failed"
                    if result.evaluation is not None and not result.evaluation.passed
                    else result.status
                )
            serialized.append(
                {
                    "task_id": message.task_id,
                    "assigned_to": message.agent_to,
                    "task_type": message.task_type,
                    "instruction": message.instruction,
                    "depends_on": list(message.depends_on),
                    "priority": message.priority,
                    "retry_count": message.retry_count,
                    "status": serialized_status,
                    "evaluation": (
                        result.evaluation.model_dump(mode="json")
                        if result is not None and result.evaluation is not None
                        else None
                    ),
                }
            )
        return serialized

    def _resolve_session_context(self, context: ResearchAgentToolContext) -> dict[str, Any]:
        """Extract session context from conversation snapshot (sync, no LLM)."""
        request_metadata = context.request.metadata if isinstance(context.request.metadata, dict) else {}
        metadata_context = (
            dict(request_metadata.get("context") or {})
            if isinstance(request_metadata.get("context"), dict)
            else {}
        )
        conversation = None
        if context.request.conversation_id:
            conversation = context.research_service.report_service.load_conversation(context.request.conversation_id)
        snapshot = conversation.snapshot if conversation is not None else None
        route_mode = str(
            metadata_context.get("route_mode")
            or (snapshot.active_route_mode if snapshot is not None else "")
            or "research_follow_up"
        ).strip() or "research_follow_up"
        active_thread_id = str(
            metadata_context.get("active_thread_id")
            or (snapshot.active_thread_id if snapshot is not None else "")
        ).strip() or None
        active_thread_topic = None
        if snapshot is not None and active_thread_id:
            for thread in snapshot.thread_history:
                if thread.thread_id == active_thread_id:
                    active_thread_topic = thread.topic or None
                    break
        previous_topic = active_thread_topic or (snapshot.topic if snapshot is not None else "")
        return {
            "route_mode": route_mode,
            "active_thread_id": active_thread_id,
            "active_thread_topic": active_thread_topic,
            "previous_topic": previous_topic,
        }

    def _state_from_context(
        self,
        context: ResearchAgentToolContext,
        *,
        trace: list[ResearchAgentTraceStep] | None = None,
        user_intent: ResearchUserIntentResult | None = None,
        session_context: dict[str, Any] | None = None,
        execution_plan: list[dict[str, Any]] | None = None,
    ) -> ResearchSupervisorState:
        task = context.task
        papers = context.papers
        importable_paper_count = sum(
            1
            for paper in papers
            if paper.pdf_url and paper.ingest_status not in {"ingested", "unavailable"}
        )
        has_import_candidates = importable_paper_count > 0
        open_todo_count = sum(1 for item in (task.todo_items if task else []) if item.status == "open")
        workspace = context.workspace
        workspace_stage = workspace.current_stage if workspace is not None else self._derive_workspace_stage(context)
        evidence_gap_count = len(workspace.evidence_gaps) if workspace is not None else len(context.report.gaps) if context.report else 0
        resolved_trace = trace or []
        failed_actions = [step.action_name for step in resolved_trace if step.status == "failed"]
        workspace_metadata = dict(workspace.metadata) if workspace is not None else {}
        research_context = (
            context.execution_context.research_context
            if context.execution_context is not None
            else None
        )
        request_metadata = context.request.metadata if isinstance(context.request.metadata, dict) else {}
        research_goal_lower = context.request.message.lower()
        session_ctx = session_context or self._resolve_session_context(context)
        metadata_comparison_dimension_values = request_metadata.get("comparison_dimensions")
        metadata_comparison_dimensions = [
            str(item).strip()
            for item in (metadata_comparison_dimension_values if isinstance(metadata_comparison_dimension_values, list) else [])
            if str(item).strip()
        ]
        comparison_dimensions = list(
            dict.fromkeys(
                [
                    *(item.strip() for item in context.request.comparison_dimensions if item.strip()),
                    *metadata_comparison_dimensions,
                ]
            )
        )
        recommendation_goal = str(
            context.request.recommendation_goal
            or request_metadata.get("recommendation_goal")
            or ""
        ).strip() or None
        recommendation_top_k = max(
            1,
            int(
                context.request.recommendation_top_k
                or request_metadata.get("recommendation_top_k")
                or 3
            ),
        )
        user_profile = context.research_service.memory_gateway.load_user_profile()
        force_context_compression = bool(
            context.request.force_context_compression
            or request_metadata.get("force_context_compression")
        )
        active_paper_ids = self._active_paper_ids_for_manager(context)
        context_compressed = bool(
            context.compressed_context_summary
            or workspace_metadata.get("context_compression")
        )
        session_history_count = len(research_context.session_history) if research_context is not None else 0
        context_size_large = False
        _exec_ctx = context.execution_context
        if _exec_ctx is not None and hasattr(_exec_ctx, "context_slices"):
            _manager_slice = (_exec_ctx.context_slices or {}).get("manager")
            if _manager_slice is not None:
                try:
                    import json as _json
                    _slice_data = _manager_slice.model_dump(mode="json") if hasattr(_manager_slice, "model_dump") else dict(_manager_slice)
                    context_size_large = len(_json.dumps(_slice_data, ensure_ascii=False, default=str)) > 80_000
                except Exception:
                    pass
        intent_flags = resolve_intent_flags(
            research_goal_lower=research_goal_lower,
            advanced_action=context.request.advanced_action,
            comparison_dimensions=comparison_dimensions,
            recommendation_goal=recommendation_goal,
            selected_paper_ids=list(context.request.selected_paper_ids),
            active_paper_ids=active_paper_ids,
            paper_count=len(papers),
            has_task=task is not None,
            has_papers=bool(task or papers),
            session_history_count=session_history_count,
            context_compressed=context_compressed,
            force_context_compression=force_context_compression,
            context_size_large=context_size_large,
        )
        paper_analysis_requested = intent_flags.paper_analysis_requested
        preference_recommendation_requested = intent_flags.preference_recommendation_requested
        analysis_focus = intent_flags.analysis_focus
        context_compression_needed = intent_flags.context_compression_needed
        route_mode = session_ctx["route_mode"]
        active_thread_id = session_ctx["active_thread_id"]
        active_thread_topic = session_ctx["active_thread_topic"]
        previous_topic = session_ctx["previous_topic"]
        if user_intent is None:
            user_intent = self.user_intent_resolver.resolve(
                message=context.request.message,
                has_task=task is not None,
                candidate_paper_count=len(papers),
                candidate_papers=self._candidate_paper_scope_for_manager(papers),
                active_paper_ids=active_paper_ids,
                selected_paper_ids=list(context.request.selected_paper_ids),
                has_visual_anchor=bool(context.request.chart_image_path or context.request.chart_id),
                has_document_input=bool(context.request.document_file_path),
                session_topic=previous_topic or None,
            )
        topic_continuity_score = self._topic_continuity_score(
            context.request.message,
            previous_topic,
        )
        new_topic_detected = bool(user_intent.is_new_topic) or (
            bool(previous_topic)
            and topic_continuity_score < 0.15
        )
        should_ignore_research_context = route_mode == "general_chat" or new_topic_detected
        return ResearchSupervisorState(
            goal=context.request.message,
            mode=context.request.mode,
            route_mode=route_mode,
            workflow_constraint=str(request_metadata.get("workflow_constraint") or "").strip() or None,
            active_thread_id=active_thread_id,
            active_thread_topic=active_thread_topic,
            topic_continuity_score=topic_continuity_score,
            new_topic_detected=new_topic_detected,
            should_ignore_research_context=should_ignore_research_context,
            task_id=context.request.task_id or (task.task_id if task else None),
            has_task=task is not None,
            has_report=context.report is not None,
            paper_count=len(papers),
            imported_document_count=len(task.imported_document_ids) if task else 0,
            has_document_input=bool(context.request.document_file_path),
            has_chart_input=bool(context.request.chart_image_path),
            document_understood=context.document_attempted or context.parsed_document is not None,
            chart_understood=context.chart_attempted or context.chart_result is not None,
            has_import_candidates=has_import_candidates,
            importable_paper_count=importable_paper_count,
            selected_paper_count=len(context.request.selected_paper_ids),
            active_paper_ids=active_paper_ids,
            auto_import=context.request.auto_import,
            import_top_k=context.request.import_top_k,
            import_attempted=context.import_attempted or context.import_result is not None,
            answer_attempted=context.qa_result is not None,
            open_todo_count=open_todo_count,
            evidence_gap_count=evidence_gap_count,
            workspace_stage=workspace_stage,
            workspace_ready=bool(task and (task.imported_document_ids or context.report is not None)),
            paper_analysis_requested=paper_analysis_requested,
            preference_recommendation_requested=preference_recommendation_requested,
            known_interest_count=len(user_profile.interest_topics or user_profile.research_interests),
            analysis_focus=analysis_focus,
            comparison_dimensions=comparison_dimensions,
            recommendation_goal=recommendation_goal,
            recommendation_top_k=recommendation_top_k,
            force_context_compression=force_context_compression,
            context_compression_needed=context_compression_needed,
            paper_analysis_completed=bool(
                context.paper_analysis_result is not None or workspace_metadata.get("latest_paper_analysis")
            ),
            context_compressed=context_compressed,
            last_action_name=resolved_trace[-1].action_name if resolved_trace else None,
            failed_actions=failed_actions,
            candidate_papers=self._candidate_paper_scope_for_manager(papers),
            user_intent=user_intent.model_dump(mode="json"),
            skill_context=context.skill_context,
            execution_plan=list(execution_plan or []),
        )

    def _topic_continuity_score(self, current: str | None, previous: str | None) -> float:
        current_tokens = set(_normalize_topic_text_impl(current).split())
        previous_tokens = set(_normalize_topic_text_impl(previous).split())
        if not current_tokens or not previous_tokens:
            return 0.0
        return len(current_tokens & previous_tokens) / max(1, len(current_tokens | previous_tokens))

    def _active_paper_ids_for_manager(self, context: ResearchAgentToolContext) -> list[str]:
        metadata = context.request.metadata if isinstance(context.request.metadata, dict) else {}
        metadata_context = metadata.get("context")
        active_ids: list[str] = []
        if isinstance(metadata_context, dict):
            raw_active = metadata_context.get("active_paper_ids")
            if isinstance(raw_active, list):
                active_ids.extend(str(item).strip() for item in raw_active if str(item).strip())
        if context.execution_context is not None:
            active_ids.extend(
                str(item).strip()
                for item in getattr(context.execution_context.research_context, "active_papers", [])
                if str(item).strip()
            )
        return self._dedupe_ids(active_ids or context.request.selected_paper_ids)

    def _candidate_paper_scope_for_manager(self, papers: list[PaperCandidate], *, limit: int = 12) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for index, paper in enumerate(papers[:limit], start=1):
            candidates.append(
                {
                    "index": index,
                    "paper_id": paper.paper_id,
                    "title": paper.title,
                    "year": paper.year,
                    "source": paper.source,
                    "relevance_score": paper.relevance_score,
                    "citations": paper.citations,
                    "has_imported_document": bool(str(paper.metadata.get("document_id") or "").strip()),
                }
            )
        return candidates

    def _build_response(
        self,
        *,
        request: ResearchAgentRunRequest,
        context: ResearchAgentToolContext,
        trace: list[ResearchAgentTraceStep],
        failed: bool,
        exhausted: bool,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        clarification_request: str | None,
        active_plan_id: str | None,
    ) -> ResearchAgentRunResponse:
        return self.result_aggregator.build_response(
            request=request,
            context=context,
            trace=trace,
            failed=failed,
            exhausted=exhausted,
            agent_messages=agent_messages,
            agent_results=agent_results,
            planner_runs=planner_runs,
            replan_count=replan_count,
            clarification_request=clarification_request,
            active_plan_id=active_plan_id,
        )

    def _log_internal_runtime_state(
        self,
        *,
        request: ResearchAgentRunRequest,
        workspace: ResearchWorkspaceState,
        trace: list[ResearchAgentTraceStep],
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        clarification_request: str | None,
    ) -> None:
        logger.info(
            "Research agent internals | mode=%s | task=%s | workspace=%s | stop_reason=%s | clarification=%s | plan=%s | trace=%s",
            request.mode,
            request.task_id,
            workspace.status_summary or workspace.current_stage,
            workspace.stop_reason,
            clarification_request or "",
            [
                {
                    "task_id": message.task_id,
                    "agent": message.agent_to,
                    "type": message.task_type,
                    "priority": message.priority,
                    "status": next(
                        (
                            result.status
                            for result in agent_results
                            if result.task_id == message.task_id
                        ),
                        "planned",
                    ),
                }
                for message in agent_messages[-8:]
            ],
            [
                {
                    "step": step.step_index,
                    "agent": step.agent,
                    "action": step.action_name,
                    "phase": step.phase,
                    "status": step.status,
                }
                for step in trace
            ],
        )

    def _build_messages(
        self,
        request: ResearchAgentRunRequest,
        context: ResearchAgentToolContext,
        trace: list[ResearchAgentTraceStep],
        workspace: ResearchWorkspaceState,
        *,
        agent_messages: list[AgentMessage],
        agent_results: list[AgentResultMessage],
        clarification_request: str | None,
        replan_count: int,
    ) -> list[ResearchMessage]:
        return self.response_formatter.build_messages(
            request,
            context,
            trace,
            workspace,
            agent_messages=agent_messages,
            agent_results=agent_results,
            clarification_request=clarification_request,
            replan_count=replan_count,
            serialize_task_plan_fn=self._serialize_task_plan,
        )

    def _next_actions(
        self,
        context: ResearchAgentToolContext,
        workspace: ResearchWorkspaceState,
        *,
        clarification_request: str | None = None,
    ) -> list[str]:
        return self.response_formatter.build_next_actions(
            context,
            workspace,
            clarification_request=clarification_request,
        )

    def _resolved_advanced_strategy(
        self,
        context: ResearchAgentToolContext,
        *,
        workspace: ResearchWorkspaceState,
    ) -> ResearchAdvancedStrategy:
        base_strategy = self._read_advanced_strategy(context.workspace or workspace)
        request_metadata = context.request.metadata if isinstance(context.request.metadata, dict) else {}
        updated = base_strategy.model_copy(deep=True)
        metadata_dimensions = request_metadata.get("comparison_dimensions")
        explicit_dimensions = [
            str(item).strip().lower()
            for item in [
                *context.request.comparison_dimensions,
                *(metadata_dimensions if isinstance(metadata_dimensions, list) else []),
            ]
            if str(item).strip()
        ]
        explicit_goal = str(
            context.request.recommendation_goal
            or request_metadata.get("recommendation_goal")
            or ""
        ).strip()
        explicit_action = context.request.advanced_action
        if explicit_action is not None:
            updated.action = explicit_action
        if explicit_action is None and (
            context.paper_analysis_result is not None or workspace.metadata.get("latest_paper_analysis")
        ):
            updated.action = "analyze"
        if explicit_dimensions:
            updated.comparison_dimensions = list(dict.fromkeys(explicit_dimensions))
        elif updated.action in {"analyze", "compare"} and not updated.comparison_dimensions:
            comparison_payload = workspace.metadata.get("latest_comparison")
            if isinstance(comparison_payload, dict):
                updated.comparison_dimensions = list(
                    dict.fromkeys(
                        str(item.get("dimension")).strip().lower()
                        for item in comparison_payload.get("table", [])
                        if isinstance(item, dict) and str(item.get("dimension") or "").strip()
                    )
                )
        if explicit_goal:
            updated.recommendation_goal = explicit_goal
        if explicit_action in {"analyze", "recommend"} or "recommendation_top_k" in request_metadata:
            updated.recommendation_top_k = max(
                1,
                min(
                    10,
                    int(
                        context.request.recommendation_top_k
                        or request_metadata.get("recommendation_top_k")
                        or updated.recommendation_top_k
                        or 3
                    ),
                ),
            )
        if (
            explicit_action in {"analyze", "compare", "recommend"}
            or "force_context_compression" in request_metadata
            or context.request.force_context_compression
        ):
            updated.force_context_compression = bool(
                context.request.force_context_compression
                or request_metadata.get("force_context_compression")
            )
        return updated

    def _read_advanced_strategy(
        self,
        workspace: ResearchWorkspaceState | None,
    ) -> ResearchAdvancedStrategy:
        if workspace is None:
            return ResearchAdvancedStrategy()
        payload = workspace.metadata.get("advanced_strategy")
        if not isinstance(payload, dict):
            return ResearchAdvancedStrategy()
        try:
            return ResearchAdvancedStrategy.model_validate(payload)
        except Exception:
            return ResearchAdvancedStrategy()

    def _derive_workspace_stage(self, context: ResearchAgentToolContext) -> str:
        if context.workspace is not None and context.workspace.current_stage:
            if (
                context.workspace.metadata.get("latest_paper_analysis")
                or context.workspace.metadata.get("latest_comparison")
                or context.workspace.metadata.get("latest_recommendations")
                or context.workspace.metadata.get("latest_preference_recommendations")
            ):
                return "complete"
            return context.workspace.current_stage
        if context.request.mode == "document" or context.parsed_document is not None:
            return "document"
        if context.request.mode == "chart" or context.chart_result is not None:
            return "chart"
        if context.qa_result is not None:
            return "qa"
        if context.preference_recommendation_result is not None:
            return "complete"
        if context.import_result is not None:
            return "ingest"
        if context.task is not None:
            return "qa" if context.task.imported_document_ids else "ingest" if context.papers else "discover"
        return "discover"

    def _workspace_summary(self, context: ResearchAgentToolContext) -> str:
        if context.workspace is not None and context.workspace.status_summary:
            return context.workspace.status_summary
        paper_count = len(context.papers)
        imported_document_count = len(context.task.imported_document_ids) if context.task else 0
        open_todo_count = sum(1 for item in (context.task.todo_items if context.task else []) if item.status == "open")
        gap_count = len(context.workspace.evidence_gaps) if context.workspace is not None else len(context.report.gaps) if context.report else 0
        return (
            f"stage={self._derive_workspace_stage(context)}; papers={paper_count}; "
            f"imported_docs={imported_document_count}; open_todos={open_todo_count}; evidence_gaps={gap_count}"
        )

    def _stop_reason_from_trace(
        self,
        trace: list[ResearchAgentTraceStep],
        *,
        failed: bool,
        exhausted: bool,
    ) -> str | None:
        for step in reversed(trace):
            stop_reason = step.metadata.get("stop_reason") if isinstance(step.metadata, dict) else None
            if isinstance(stop_reason, str) and stop_reason.strip():
                return stop_reason
        if failed:
            return "The manager stopped because a tool action failed and returned a partial workspace."
        if exhausted:
            return f"The manager reached max_steps={self.max_steps} before completing the research loop."
        return None

    def _workspace_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.__class__.__name__,
            "manager_engine": "llm_dynamic_decision_loop",
            "routing_engine": "llm_native_multi_agent",
        }

    def _response_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.__class__.__name__,
            "manager_agent": self._manager_display_name(),
            "manager_engine": "llm_dynamic_decision_loop",
            "routing_engine": "llm_native_multi_agent",
            "autonomy_mode": "manager_delegates_to_worker_agents",
            "agent_architecture": "llm_native_manager_worker_agents",
            "primary_agents": [
                "ResearchSupervisorAgent",
                "GeneralAnswerAgent",
                "ChartAnalysisAgent",
                "LiteratureScoutAgent",
                "ResearchKnowledgeAgent",
                "ResearchWriterAgent",
                "PaperAnalysisAgent",
                "PreferenceMemoryAgent",
            ],
            "primary_runtime_workers": [],
            "primary_skills": [
                "PaperCurator",
                "TopicPlanner",
                "ResearchQueryRewriter",
                "PaperRanker",
                "SurveyWriter",
                "PaperAnalyzer",
            ],
        }

    def _resolve_workspace(
        self,
        *,
        context: ResearchAgentToolContext,
        trace: list[ResearchAgentTraceStep],
        failed: bool,
        exhausted: bool,
    ) -> ResearchWorkspaceState:
        stage = self._derive_workspace_stage(context)
        stop_reason = self._stop_reason_from_trace(trace, failed=failed, exhausted=exhausted) or (
            context.workspace.stop_reason if context.workspace is not None else None
        )
        metadata = {**self._workspace_runtime_metadata(), "trace_steps": len(trace)}
        extra_questions = [context.request.message] if context.request.message else []
        extra_findings = [context.qa_result.qa.answer] if context.qa_result is not None else []
        if context.preference_recommendation_result is not None:
            metadata["latest_preference_recommendations"] = context.preference_recommendation_result.model_dump(mode="json")
        if context.workspace is not None:
            next_actions = list(context.workspace.next_actions)
            if stop_reason:
                stop_action = f"Stop reason: {stop_reason}"
                if stop_action not in next_actions:
                    next_actions.append(stop_action)
            return context.workspace.model_copy(
                update={
                    "current_stage": stage,
                    "stop_reason": stop_reason,
                    "status_summary": context.workspace.status_summary or self._workspace_summary(context),
                    "next_actions": next_actions[:4],
                    "metadata": {**context.workspace.metadata, **metadata},
                }
            )
        if context.task is not None or context.report is not None:
            return build_workspace_from_task(
                task=context.task,
                report=context.report,
                papers=context.papers,
                extra_questions=extra_questions,
                extra_findings=extra_findings,
                stage=stage,  # type: ignore[arg-type]
                stop_reason=stop_reason,
                metadata=metadata,
            )
        return build_workspace_state(
            objective=context.request.message,
            stage=stage,  # type: ignore[arg-type]
            papers=context.papers,
            imported_document_ids=context.task.imported_document_ids if context.task else [],
            report=context.report,
            todo_items=context.task.todo_items if context.task else [],
            extra_questions=extra_questions,
            extra_findings=extra_findings,
            stop_reason=stop_reason,
            metadata=metadata,
        )


class ResearchSupervisorGraphRuntime(ResearchRuntimeBase):
    """High-level supervisor runtime backed by a LangGraph loop."""

    def __init__(self, **kwargs: Any) -> None:
        from langgraph.graph import END, StateGraph

        super().__init__(**kwargs)
        graph = StateGraph(ResearchAgentGraphState)
        graph.add_node("bootstrap_context_node", self.bootstrap_context_node)
        graph.add_node("supervisor_node", self.supervisor_node)
        graph.add_node("literature_scout_node", self.literature_scout_node)
        graph.add_node("write_review_node", self.write_review_specialist_node)
        graph.add_node("paper_import_node", self.paper_import_specialist_node)
        graph.add_node("zotero_sync_node", self.zotero_sync_specialist_node)
        graph.add_node("research_qa_node", self.research_qa_specialist_node)
        graph.add_node("general_answer_node", self.general_answer_specialist_node)
        graph.add_node("preference_memory_node", self.preference_memory_specialist_node)
        graph.add_node("paper_analysis_node", self.paper_analysis_specialist_node)
        graph.add_node("context_compression_node", self.context_compression_specialist_node)
        graph.add_node("document_specialist_node", self.document_specialist_node)
        graph.add_node("chart_specialist_node", self.chart_specialist_node)
        graph.add_node("paper_figure_analysis_node", self.paper_figure_analysis_specialist_node)
        graph.add_node("finalize_node", self.finalize_node)
        graph.add_edge("__start__", "bootstrap_context_node")
        graph.add_edge("finalize_node", END)
        self.graph = graph.compile()

    async def run(self, request: ResearchAgentRunRequest, *, graph_runtime: Any, on_progress: Any | None = None) -> ResearchAgentRunResponse:
        context = await self._build_tool_context(request=request, graph_runtime=graph_runtime)
        if on_progress is not None:
            context.progress_callback = on_progress
        _update_runtime_progress(
            context,
            stage="supervisor_graph",
            node="bootstrap_context_node",
            status="started",
            summary="Supervisor graph initialized.",
        )
        initial_state: ResearchAgentGraphState = {
            "context": context,
            "trace": [],
            "failed": False,
            "exhausted": False,
            "current_step_index": 0,
            "pending_agent_messages": [],
            "agent_messages": [],
            "agent_results": [],
            "completed_agent_task_ids": [],
            "failed_agent_task_ids": [],
            "replanned_failure_task_ids": [],
            "planner_runs": 0,
            "replan_count": 0,
            "clarification_request": None,
            "active_plan_id": None,
            "execution_plan": [],
            "progress_signature": "",
            "stagnant_decision_count": 0,
            "repeated_action_count": 0,
        }
        heartbeat_task = asyncio.create_task(self._heartbeat_runtime_progress(context))
        try:
            graph_state = await self.graph.ainvoke(initial_state)
        finally:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        if graph_state.get("exhausted"):
            context.warnings.append(f"research supervisor graph reached max_steps={self.max_steps} before finalize")
        return self._build_response(
            request=request,
            context=context,
            trace=graph_state.get("trace", []),
            failed=bool(graph_state.get("failed")),
            exhausted=bool(graph_state.get("exhausted")),
            agent_messages=graph_state.get("agent_messages", []),
            agent_results=graph_state.get("agent_results", []),
            planner_runs=int(graph_state.get("planner_runs", 0) or 0),
            replan_count=int(graph_state.get("replan_count", 0) or 0),
            clarification_request=graph_state.get("clarification_request"),
            active_plan_id=graph_state.get("active_plan_id"),
        )

    async def _heartbeat_runtime_progress(self, context: ResearchAgentToolContext) -> None:
        while True:
            await asyncio.sleep(5.0)
            progress = dict(context.runtime_progress or {})
            if not progress:
                continue
            context.research_service.append_runtime_event(
                conversation_id=context.request.conversation_id,
                event_type="memory_updated",
                task_id=context.task.task_id if context.task is not None else context.request.task_id,
                correlation_id=(
                    context.task.status_metadata.correlation_id
                    if context.task is not None
                    else None
                ),
                payload={
                    "runtime_event": "supervisor_heartbeat",
                    **progress,
                },
            )

    async def _decide_next_action(self, state: ResearchAgentGraphState) -> ResearchSupervisorDecision:
        decision = await super()._decide_next_action(state)
        metadata = dict(decision.metadata)
        if metadata.get("worker_agent") == "ResearchSupervisorAgent":
            metadata["worker_agent"] = self._manager_trace_agent_name()
        return ResearchSupervisorDecision(
            action_name=decision.action_name,
            thought=decision.thought,
            rationale=decision.rationale,
            phase=decision.phase,
            estimated_gain=decision.estimated_gain,
            estimated_cost=decision.estimated_cost,
            stop_reason=decision.stop_reason,
            action_input=dict(decision.action_input),
            metadata=metadata,
        )

    def _workspace_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.__class__.__name__,
            "manager_engine": "langgraph_supervisor",
            "routing_engine": "langgraph_supervisor_specialists",
        }

    def _manager_display_name(self) -> str:
        return "ResearchSupervisorAgent"

    def _manager_trace_agent_name(self) -> str:
        return "ResearchSupervisorAgent"

    def _response_runtime_metadata(self) -> dict[str, Any]:
        return {
            "runtime": self.__class__.__name__,
            "manager_agent": self._manager_display_name(),
            "manager_engine": "langgraph_supervisor",
            "routing_engine": "langgraph_supervisor_specialists",
            "autonomy_mode": "supervisor_delegates_to_specialists",
            "agent_architecture": "langgraph_supervisor_multi_agent",
            "primary_agents": [
                "ResearchSupervisorAgent",
                "GeneralAnswerAgent",
                "ChartAnalysisAgent",
                "LiteratureScoutAgent",
                "ResearchKnowledgeAgent",
                "ResearchWriterAgent",
                "PaperAnalysisAgent",
                "PreferenceMemoryAgent",
            ],
            "primary_runtime_workers": [],
            "primary_skills": [
                "PaperCurator",
                "TopicPlanner",
                "ResearchQueryRewriter",
                "PaperRanker",
                "SurveyWriter",
                "PaperAnalyzer",
            ],
            "supervisor_graph": {
                "entry_node": "bootstrap_context_node",
                "decision_node": "supervisor_node",
                "specialist_nodes": [
                    "literature_scout_node",
                    "write_review_node",
                    "paper_import_node",
                    "zotero_sync_node",
                    "research_qa_node",
                    "general_answer_node",
                    "preference_memory_node",
                    "paper_analysis_node",
                    "context_compression_node",
                    "document_specialist_node",
                    "chart_specialist_node",
                    "paper_figure_analysis_node",
                ],
                "finalize_node": "finalize_node",
            },
        }

    async def bootstrap_context_node(self, state: ResearchAgentGraphState):
        from langgraph.types import Command

        _update_runtime_progress(
            state["context"],
            stage="supervisor_graph",
            node="bootstrap_context_node",
            status="running",
            summary="Bootstrapping supervisor graph context.",
        )

        return Command(
            update={
                "progress_signature": self._progress_signature(state),
                "trace": list(state.get("trace", [])),
            },
            goto="supervisor_node",
        )

    async def supervisor_node(self, state: ResearchAgentGraphState):
        from langgraph.types import Command

        _update_runtime_progress(
            state["context"],
            stage="supervisor_graph",
            node="supervisor_node",
            status="running",
            summary="Manager is choosing the next action.",
            extra={"step_index": int(state.get("current_step_index", 0) or 0) + 1},
        )
        current_step_index = int(state.get("current_step_index", 0) or 0)
        if current_step_index >= self.max_steps:
            return Command(update={"exhausted": True}, goto="finalize_node")
        if self._should_force_finalize(state):
            decision = self._forced_finalize_decision(state)
            update = self._on_decision(state, current_step_index + 1, decision)
            return Command(
                update={
                    "current_step_index": current_step_index + 1,
                    "current_decision": decision,
                    **update,
                },
                goto="finalize_node",
            )
        step_index = current_step_index + 1
        decision = await self._decide_next_action(state)
        update = self._on_decision(state, step_index, decision)
        goto = self._goto_after_supervisor(decision)
        return Command(
            update={
                "current_step_index": step_index,
                "current_decision": decision,
                **update,
            },
            goto=goto,
        )

    async def literature_scout_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.search_literature_node)

    async def write_review_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.write_review_node)

    async def paper_import_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.import_papers_node)

    async def zotero_sync_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.sync_to_zotero_node)

    async def research_qa_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.answer_question_node)

    async def general_answer_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.general_answer_node)

    async def preference_memory_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.recommend_from_preferences_node)

    async def paper_analysis_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.analyze_papers_node)

    async def context_compression_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.compress_context_node)

    async def document_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.understand_document_node)

    async def chart_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.understand_chart_node)

    async def paper_figure_analysis_specialist_node(self, state: ResearchAgentGraphState):
        return await self._run_specialist_and_route(state, self.analyze_paper_figures_node)

    async def _run_specialist_and_route(self, state: ResearchAgentGraphState, runner):
        from langgraph.types import Command

        update = await runner(state)
        next_node = "finalize_node" if self._should_force_finalize({**state, **update}) else "supervisor_node"
        return Command(update=update, goto=next_node)

    def _goto_after_supervisor(self, decision: ResearchSupervisorDecision) -> str:
        action = self._route_decision(decision)
        mapping = {
            "search_literature": "literature_scout_node",
            "write_review": "write_review_node",
            "import_papers": "paper_import_node",
            "sync_to_zotero": "zotero_sync_node",
            "answer_question": "research_qa_node",
            "general_answer": "general_answer_node",
            "recommend_from_preferences": "preference_memory_node",
            "analyze_papers": "paper_analysis_node",
            "compress_context": "context_compression_node",
            "understand_document": "document_specialist_node",
            "supervisor_understand_chart": "chart_specialist_node",
            "analyze_paper_figures": "paper_figure_analysis_node",
            "finalize": "finalize_node",
        }
        return mapping[action]
