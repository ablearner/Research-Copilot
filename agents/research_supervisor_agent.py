from __future__ import annotations

from dataclasses import dataclass, field, replace
import logging
import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from adapters.llm.base import BaseLLMAdapter, LLMAdapterError
from domain.schemas.agent_message import AgentMessage, AgentResultMessage
from domain.schemas.research_context import ResearchContextSlice
from domain.schemas.sub_manager import TaskEvaluation
from tools.research import ResearchEvaluator


logger = logging.getLogger(__name__)


ResearchSupervisorActionName = Literal[
    "clarify_request",
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
    "finalize",
]


class PlanStep(BaseModel):
    """A single step in a multi-step execution plan."""
    step_id: int
    action: str
    instruction: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"


@dataclass(slots=True)
class ResearchSupervisorState:
    goal: str
    mode: str = "auto"
    route_mode: str = "research_follow_up"
    workflow_constraint: str | None = None
    active_thread_id: str | None = None
    active_thread_topic: str | None = None
    topic_continuity_score: float = 0.0
    new_topic_detected: bool = False
    should_ignore_research_context: bool = False
    task_id: str | None = None
    has_task: bool = False
    has_report: bool = False
    paper_count: int = 0
    imported_document_count: int = 0
    has_document_input: bool = False
    has_chart_input: bool = False
    document_understood: bool = False
    chart_understood: bool = False
    has_import_candidates: bool = False
    importable_paper_count: int = 0
    selected_paper_count: int = 0
    active_paper_ids: list[str] = field(default_factory=list)
    auto_import: bool = True
    import_top_k: int = 3
    import_attempted: bool = False
    answer_attempted: bool = False
    open_todo_count: int = 0
    evidence_gap_count: int = 0
    workspace_stage: str = "discover"
    workspace_ready: bool = False
    paper_analysis_requested: bool = False
    preference_recommendation_requested: bool = False
    known_interest_count: int = 0
    analysis_focus: str | None = None
    comparison_dimensions: list[str] = field(default_factory=list)
    recommendation_goal: str | None = None
    recommendation_top_k: int = 3
    force_context_compression: bool = False
    context_compression_needed: bool = False
    paper_analysis_completed: bool = False
    context_compressed: bool = False
    last_action_name: str | None = None
    failed_actions: list[str] = field(default_factory=list)
    latest_result_task_type: str | None = None
    latest_result_status: str | None = None
    latest_progress_made: bool | None = None
    latest_result_confidence: float | None = None
    latest_missing_inputs: list[str] = field(default_factory=list)
    latest_suggested_next_actions: list[str] = field(default_factory=list)
    candidate_papers: list[dict[str, Any]] = field(default_factory=list)
    user_intent: dict[str, Any] = field(default_factory=dict)
    skill_context: str | None = None
    execution_plan: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ResearchSupervisorDecision:
    action_name: ResearchSupervisorActionName
    thought: str
    rationale: str
    phase: str = "plan"
    estimated_gain: float = 0.0
    estimated_cost: float = 0.0
    stop_reason: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchSupervisorLLMDecision(BaseModel):
    resolved_intent: str = ""
    resolved_paper_ids: list[str] = Field(default_factory=list)
    action_name: str
    worker_agent: str | None = None
    instruction: str = ""
    thought: str = ""
    rationale: str = ""
    phase: Literal["plan", "act", "reflect", "commit"] = "act"
    estimated_gain: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_cost: float = Field(default=0.3, ge=0.0, le=1.0)
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    payload: dict[str, Any] = Field(default_factory=dict)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    stop_reason: str | None = None
    plan: list[PlanStep] = Field(default_factory=list)


class ResearchSupervisorAgent:
    """Goal-directed manager for the literature research assistant.

    This agent owns the "what next?" decision. The runtime executes tools and
    feeds observations back into this compact state, so the flow is an agent
    loop instead of a fixed LangGraph topology.
    """

    def __init__(
        self,
        *,
        evaluation_skill: ResearchEvaluator | None = None,
        llm_adapter: BaseLLMAdapter | None = None,
    ) -> None:
        self.evaluation_skill = evaluation_skill or ResearchEvaluator()
        self.llm_adapter = llm_adapter

    async def decide_next_action_async(
        self,
        state: ResearchSupervisorState,
        *,
        pending_messages: list[AgentMessage] | None = None,
        agent_messages: list[AgentMessage] | None = None,
        agent_results: list[AgentResultMessage] | None = None,
        completed_task_ids: list[str] | None = None,
        failed_task_ids: list[str] | None = None,
        replanned_failure_task_ids: list[str] | None = None,
        planner_runs: int = 0,
        replan_count: int = 0,
        context_slice: ResearchContextSlice | None = None,
        clarification_request: str | None = None,
        active_plan_id: str | None = None,
    ) -> ResearchSupervisorDecision:
        if self.llm_adapter is None:
            logger.warning("ResearchSupervisorAgent has no LLM adapter; refusing rule-based orchestration")
            return self._llm_unavailable_decision(
                state,
                pending_messages=pending_messages,
                agent_messages=agent_messages,
                agent_results=agent_results,
                completed_task_ids=completed_task_ids,
                failed_task_ids=failed_task_ids,
                replanned_failure_task_ids=replanned_failure_task_ids,
                planner_runs=planner_runs,
                replan_count=replan_count,
                clarification_request=clarification_request,
                active_plan_id=active_plan_id,
            )

        try:
            return await self._decide_with_llm(
                state,
                pending_messages=pending_messages,
                agent_messages=agent_messages,
                agent_results=agent_results,
                completed_task_ids=completed_task_ids,
                failed_task_ids=failed_task_ids,
                planner_runs=planner_runs,
                replan_count=replan_count,
                context_slice=context_slice,
            )
        except (LLMAdapterError, ValueError) as exc:
            logger.warning("LLM manager decision failed; stopping instead of rule-based fallback: %s", exc)
            return self._llm_unavailable_decision(
                state,
                pending_messages=pending_messages,
                agent_messages=agent_messages,
                agent_results=agent_results,
                completed_task_ids=completed_task_ids,
                failed_task_ids=failed_task_ids,
                replanned_failure_task_ids=replanned_failure_task_ids,
                planner_runs=planner_runs,
                replan_count=replan_count,
                clarification_request=clarification_request,
                active_plan_id=active_plan_id,
                reason=f"LLM manager decision failed: {exc}",
            )

    async def _decide_with_llm(
        self,
        state: ResearchSupervisorState,
        *,
        pending_messages: list[AgentMessage] | None,
        agent_messages: list[AgentMessage] | None,
        agent_results: list[AgentResultMessage] | None,
        completed_task_ids: list[str] | None,
        failed_task_ids: list[str] | None,
        planner_runs: int,
        replan_count: int,
        context_slice: ResearchContextSlice | None,
    ) -> ResearchSupervisorDecision:
        assert self.llm_adapter is not None
        all_messages = list(agent_messages or [])
        results = self._evaluate_results(results=list(agent_results or []), agent_messages=all_messages)
        state = self._state_with_recent_result_signal(state, results=results)
        context_oversize = self._context_exceeds_budget(context_slice)
        if context_oversize:
            already_compressed = any(
                r.task_type == "compress_context" for r in results
            )
            if not already_compressed and not state.context_compressed:
                return self._guardrail_worker_action(
                    action_name="compress_context",
                    state=state,
                    all_messages=all_messages,
                    results=results,
                    planner_runs=planner_runs,
                    replan_count=replan_count,
                    thought="Context slice is too large for the LLM decision call; compressing now.",
                    rationale="The serialized context exceeds the safe token budget. Running compress_context will shrink it so the supervisor LLM can make decisions.",
                    phase="act",
                    estimated_gain=0.7,
                    estimated_cost=0.1,
                    priority="high",
                )
            context_slice = self._truncate_context_slice(context_slice)
        # ── Plan-and-Execute: advance active plan ──
        plan_decision = self._advance_plan(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
            context_slice=context_slice,
        )
        if plan_decision is not None:
            return plan_decision
        guardrail_hints: list[dict[str, Any]] = []
        intent_guardrail = self._intent_guardrail_decision(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
        )
        if intent_guardrail is not None:
            guardrail_hints.append({
                "source": "intent_guardrail",
                "suggested_action": intent_guardrail.action_name,
                "thought": intent_guardrail.thought,
                "rationale": intent_guardrail.rationale,
            })
        constrained_decision = self._workflow_constraint_decision(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
        )
        if constrained_decision is not None:
            guardrail_hints.append({
                "source": "workflow_constraint",
                "suggested_action": constrained_decision.action_name,
                "thought": constrained_decision.thought,
                "rationale": constrained_decision.rationale,
            })
        guardrail_decision = self._guardrail_decision(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
        )
        if guardrail_decision is not None:
            guardrail_hints.append({
                "source": "post_execution_guardrail",
                "suggested_action": guardrail_decision.action_name,
                "thought": guardrail_decision.thought,
                "rationale": guardrail_decision.rationale,
            })
        available_actions = self._available_actions(state)
        input_data: dict[str, Any] = {
                "state": self._state_snapshot(state),
                "available_actions": available_actions,
                "recent_tasks": [self._message_snapshot(message) for message in all_messages[-8:]],
                "recent_results": [self._result_snapshot(result) for result in results[-8:]],
                "pending_tasks": [self._message_snapshot(message) for message in (pending_messages or [])[-4:]],
                "completed_task_ids": self._dedupe_ids(completed_task_ids or []),
                "failed_task_ids": self._dedupe_ids(failed_task_ids or []),
                "planner_runs": planner_runs,
                "replan_count": replan_count,
                "context_slice": self._serialize_context_slice(context_slice),
                "guardrail_hints": guardrail_hints,
        }
        if state.skill_context:
            input_data["active_skill_instructions"] = state.skill_context
        try:
            llm_output = await self.llm_adapter.generate_structured(
                prompt=self._llm_prompt(),
                input_data=input_data,
                response_model=ResearchSupervisorLLMDecision,
            )
        except (LLMAdapterError, ValueError):
            for _gr in (intent_guardrail, constrained_decision, guardrail_decision):
                if _gr is not None:
                    return _gr
            raise
        return self._decision_from_llm_output(
            llm_output=llm_output,
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
            context_slice=context_slice,
        )

    def _intent_guardrail_decision(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
    ) -> ResearchSupervisorDecision | None:
        intent_name = str(state.user_intent.get("intent") or "").strip()
        # ── Physical-input guardrails only ──
        # Intent-based routing is handled by the Supervisor LLM, which
        # receives state.user_intent as a hint.  Only physical attachment
        # checks (chart image present) are kept here because they are
        # unambiguous and independent of keyword classification.
        if (state.has_chart_input or state.mode == "chart") and not state.chart_understood:
            return self._guardrail_worker_action(
                action_name="supervisor_understand_chart",
                state=state,
                all_messages=all_messages,
                results=results,
                planner_runs=planner_runs,
                replan_count=replan_count,
                thought=(
                    "The request includes an explicit chart input, so the manager should first "
                    "route it to the chart specialist before considering clarification or finalization."
                ),
                rationale=(
                    "A concrete chart image is already available; asking for another paper or chart "
                    "identifier would bypass the provided visual evidence."
                ),
                phase="act",
                estimated_gain=0.9,
                estimated_cost=0.18,
                payload_overrides={
                    "trigger": "chart_input_guardrail",
                    "intent": intent_name or "chart_qa",
                },
                priority="high",
            )
        if (state.has_chart_input or state.mode == "chart") and state.chart_understood:
            return self._decision(
                action_name="finalize",
                thought="The provided chart has already been understood by the chart specialist.",
                rationale=(
                    "The visual evidence was processed successfully, so the manager should stop without "
                    "turning the stale figure-clarification intent into a user-facing warning."
                ),
                phase="commit",
                estimated_gain=0.0,
                estimated_cost=0.0,
                stop_reason="Chart understanding completed.",
                metadata={
                    "decision_source": "manager_guardrail",
                    "worker_agent": "ResearchSupervisorAgent",
                    "state_update": {
                        "pending_agent_messages": [],
                        "agent_messages": all_messages,
                        "agent_results": results,
                        "completed_agent_task_ids": [
                            result.task_id for result in results if result.status in {"succeeded", "skipped"}
                        ],
                        "failed_agent_task_ids": [result.task_id for result in results if result.status == "failed"],
                        "replanned_failure_task_ids": [],
                        "planner_runs": planner_runs,
                        "replan_count": replan_count,
                        "clarification_request": None,
                        "active_plan_id": None,
                        "new_topic_detected": state.new_topic_detected,
                    },
                },
            )
        return None

    def _advance_plan(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        context_slice: ResearchContextSlice | None,
    ) -> ResearchSupervisorDecision | None:
        """If an execution plan is active, advance to the next pending step.

        Returns ``None`` when there is no plan, the plan is finished, or the
        previous step failed (which clears the plan so the LLM can replan).
        """
        plan = state.execution_plan
        if not plan:
            return None

        # Mark the previously running step based on latest result
        latest = results[-1] if results else None
        for step in plan:
            if step.get("status") == "running":
                if latest and latest.status in ("succeeded", "skipped"):
                    step["status"] = "done"
                elif latest and latest.status == "failed":
                    step["status"] = "failed"
                    # Clear the plan so the LLM can replan remaining steps
                    logger.info(
                        "Plan step %s (%s) failed; clearing plan for replan",
                        step.get("step_id"),
                        step.get("action"),
                    )
                    state.execution_plan = []
                    return None

        # Find the next pending step
        next_step: dict[str, Any] | None = None
        for step in plan:
            if step.get("status") == "pending":
                # Check dependencies are satisfied
                deps = step.get("depends_on", [])
                all_deps_done = all(
                    any(
                        s.get("step_id") == dep_id and s.get("status") == "done"
                        for s in plan
                    )
                    for dep_id in deps
                )
                if all_deps_done:
                    next_step = step
                    break

        if next_step is None:
            # All steps done or no eligible step — clear plan
            state.execution_plan = []
            return None

        # Mark step as running
        next_step["status"] = "running"
        action_name = self._normalize_action_name(next_step.get("action", "finalize"))
        instruction = next_step.get("instruction", "")
        payload = dict(next_step.get("params") or {})
        step_index = next_step.get("step_id", 0)
        total_steps = len(plan)
        remaining = sum(1 for s in plan if s.get("status") == "pending")

        return self._guardrail_worker_action(
            action_name=action_name,
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
            thought=f"Executing plan step {step_index}/{total_steps}: {action_name}. {remaining} steps remaining.",
            rationale=instruction or f"Plan step {step_index} requires {action_name}.",
            phase="act",
            estimated_gain=0.8,
            estimated_cost=0.15,
            payload_overrides={
                "trigger": "plan_execution",
                "plan_step_id": step_index,
                "plan_total_steps": total_steps,
                **payload,
            },
            priority="high",
        )

    def _workflow_constraint_decision(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
    ) -> ResearchSupervisorDecision | None:
        if state.workflow_constraint != "discovery_only":
            return None
        if state.last_action_name == "search_literature":
            if state.latest_result_status == "succeeded" and (state.has_report or state.paper_count > 0):
                return self._guardrail_finalize(
                    state,
                    all_messages=all_messages,
                    results=results,
                    planner_runs=planner_runs,
                    replan_count=replan_count,
                    stop_reason="Discovery-only workflow completed after literature search.",
                )
            if state.latest_result_status == "failed":
                return self._guardrail_finalize(
                    state,
                    all_messages=all_messages,
                    results=results,
                    planner_runs=planner_runs,
                    replan_count=replan_count,
                    stop_reason="Discovery-only workflow stopped after literature search failed.",
                )
        return self._guardrail_worker_action(
            action_name="search_literature",
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
            thought="This workflow is constrained to a single discovery pass, so the manager should execute literature search and stop.",
            rationale="Discovery-only entry points should not branch into review drafting, import, or QA after the search result is produced.",
            phase="act",
            estimated_gain=0.96,
            estimated_cost=0.18,
            payload_overrides={
                "trigger": "workflow_constraint",
                "workflow_constraint": "discovery_only",
            },
            priority="high",
        )

    def _decision_from_llm_output(
        self,
        *,
        llm_output: ResearchSupervisorLLMDecision,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        context_slice: ResearchContextSlice | None,
    ) -> ResearchSupervisorDecision:
        action_name = self._normalize_action_name(llm_output.action_name)
        thought = llm_output.thought.strip() or f"Manager selected {action_name} as the next best step."
        rationale = llm_output.rationale.strip() or "The manager chose the worker that can make the most progress."
        # ── Plan-and-Execute: store multi-step plan ──
        execution_plan: list[dict[str, Any]] = []
        if len(llm_output.plan) > 1:
            execution_plan = [step.model_dump() for step in llm_output.plan]
            # Mark the first step as running (it's the action_name the LLM chose)
            if execution_plan:
                execution_plan[0]["status"] = "running"
            logger.info(
                "LLM generated %d-step plan: %s",
                len(execution_plan),
                " → ".join(s.get("action", "?") for s in execution_plan),
            )
        metadata: dict[str, Any] = {
            "decision_source": "llm",
            "state_update": {
                "pending_agent_messages": [],
                "agent_messages": all_messages,
                "agent_results": results,
                "completed_agent_task_ids": [result.task_id for result in results if result.status in {"succeeded", "skipped"}],
                "failed_agent_task_ids": [result.task_id for result in results if result.status == "failed"],
                "replanned_failure_task_ids": [],
                "planner_runs": planner_runs + 1,
                "replan_count": replan_count,
                "clarification_request": None,
                "active_plan_id": None,
                "new_topic_detected": state.new_topic_detected,
                "execution_plan": execution_plan,
            },
        }
        if action_name == "finalize":
            stop_reason = (llm_output.stop_reason or self._stop_reason(state)).strip()
            metadata["worker_agent"] = "ResearchSupervisorAgent"
            return self._decision(
                action_name="finalize",
                thought=thought,
                rationale=rationale,
                phase=llm_output.phase,
                estimated_gain=llm_output.estimated_gain,
                estimated_cost=llm_output.estimated_cost,
                stop_reason=stop_reason,
                metadata=metadata,
            )
        if action_name == "clarify_request":
            clarification = (
                str(llm_output.payload.get("clarification_question") or llm_output.stop_reason or "").strip()
                or str(state.user_intent.get("clarification_question") or self._stop_reason(state)).strip()
            )
            metadata["worker_agent"] = "ResearchSupervisorAgent"
            metadata["clarification_request"] = clarification
            metadata["state_update"]["clarification_request"] = clarification
            return self._decision(
                action_name="clarify_request",
                thought=thought,
                rationale=rationale,
                phase=llm_output.phase,
                estimated_gain=llm_output.estimated_gain,
                estimated_cost=llm_output.estimated_cost,
                action_input={
                    "clarification_question": clarification,
                    "route_mode": state.route_mode,
                },
                stop_reason=clarification,
                metadata=metadata,
            )

        worker_agent = self._worker_for_action(action_name, llm_output.worker_agent)
        plan_id = f"llm_plan_{uuid4().hex[:12]}"
        task_type = self._task_type_for_action(action_name)
        if llm_output.resolved_paper_ids:
            existing = list(state.user_intent.get("resolved_paper_ids") or [])
            merged = self._dedupe_ids(llm_output.resolved_paper_ids + existing)
            state.user_intent["resolved_paper_ids"] = merged
        payload = {
            **self._default_payload_for_action(action_name, state),
            **dict(llm_output.payload),
        }
        payload = self._normalize_payload_paper_scope(action_name=action_name, payload=payload, state=state)
        payload = self._normalize_supervisor_route_payload(action_name=action_name, payload=payload, state=state)
        instruction = llm_output.instruction.strip() or self._default_instruction_for_action(action_name, state, payload)
        active_message = AgentMessage(
            task_id=f"llm_task_{uuid4().hex[:12]}",
            agent_from="ResearchSupervisorAgent",
            agent_to=worker_agent,
            task_type=task_type,
            instruction=instruction,
            payload=payload,
            context_slice=context_slice or {},
            priority=llm_output.priority,
            expected_output_schema=llm_output.expected_output_schema or self._expected_output_schema_for_action(action_name),
            metadata={
                "plan_id": plan_id,
                "decision_source": "llm",
                "manager_action": action_name,
            },
        )
        agent_messages = [*all_messages, active_message]
        metadata.update(
            {
                "worker_agent": worker_agent,
                "worker_task_type": task_type,
                "plan_id": plan_id,
                "active_message": active_message,
            }
        )
        metadata["state_update"] = {
            **metadata["state_update"],
            "pending_agent_messages": [active_message],
            "agent_messages": agent_messages,
            "active_plan_id": plan_id,
        }
        return self._decision(
            action_name=action_name,
            thought=thought,
            rationale=rationale,
            phase=llm_output.phase,
            estimated_gain=llm_output.estimated_gain,
            estimated_cost=llm_output.estimated_cost,
            action_input={
                **payload,
                "instruction": instruction,
            },
            metadata=metadata,
        )

    def _normalize_payload_paper_scope(
        self,
        *,
        action_name: ResearchSupervisorActionName,
        payload: dict[str, Any],
        state: ResearchSupervisorState,
    ) -> dict[str, Any]:
        if action_name not in {"answer_question", "analyze_papers", "analyze_paper_figures", "import_papers", "sync_to_zotero", "compress_context"}:
            return payload
        intent_resolved_paper_ids = [
            str(item).strip()
            for item in (state.user_intent.get("resolved_paper_ids") or [])
            if str(item).strip()
        ]
        raw_values = payload.get("paper_ids")
        if not isinstance(raw_values, list) or not raw_values:
            if action_name == "analyze_papers" and state.preference_recommendation_requested:
                return payload
            if intent_resolved_paper_ids:
                return {
                    **payload,
                    "paper_ids": self._dedupe_ids(intent_resolved_paper_ids),
                    "paper_scope_source": "user_intent_resolver",
                }
            if state.active_paper_ids and action_name in {"answer_question", "analyze_papers", "compress_context"}:
                return {
                    **payload,
                    "paper_ids": list(dict.fromkeys(state.active_paper_ids)),
                    "paper_scope_source": "active_short_term_memory",
                }
            return payload
        if intent_resolved_paper_ids:
            allowed_ids = set(intent_resolved_paper_ids)
            normalized_raw_values = [str(item).strip() for item in raw_values if str(item).strip()]
            explicit_matches = [item for item in normalized_raw_values if item in allowed_ids]
            if explicit_matches:
                dropped_refs = [item for item in normalized_raw_values if item not in allowed_ids]
                normalized_payload = {
                    **payload,
                    "paper_ids": self._dedupe_ids(explicit_matches),
                    "paper_scope_source": "user_intent_resolver",
                }
                if dropped_refs:
                    normalized_payload["dropped_paper_refs"] = dropped_refs
                return normalized_payload
        by_id = {
            str(item.get("paper_id") or "").strip(): str(item.get("paper_id") or "").strip()
            for item in state.candidate_papers
            if str(item.get("paper_id") or "").strip()
        }
        by_index = {
            str(item.get("index") or "").strip(): str(item.get("paper_id") or "").strip()
            for item in state.candidate_papers
            if str(item.get("index") or "").strip() and str(item.get("paper_id") or "").strip()
        }
        ordinal_refs = {
            **{f"p{index}": paper_id for index, paper_id in enumerate(by_index.values(), start=1)},
            **{f"paper {index}": paper_id for index, paper_id in enumerate(by_index.values(), start=1)},
            **{f"第{index}篇": paper_id for index, paper_id in enumerate(by_index.values(), start=1)},
            **{f"第 {index} 篇": paper_id for index, paper_id in enumerate(by_index.values(), start=1)},
        }
        by_title = {
            self._normalize_text(str(item.get("title") or "")): str(item.get("paper_id") or "").strip()
            for item in state.candidate_papers
            if str(item.get("title") or "").strip() and str(item.get("paper_id") or "").strip()
        }
        resolved: list[str] = []
        unresolved: list[str] = []
        for raw in raw_values:
            value = str(raw).strip()
            if not value:
                continue
            normalized_value = self._normalize_text(value)
            paper_id = by_id.get(value) or by_index.get(value) or ordinal_refs.get(normalized_value) or by_title.get(normalized_value)
            if not paper_id and normalized_value:
                paper_id = self._best_title_match(normalized_value, by_title)
            if paper_id:
                resolved.append(paper_id)
            else:
                unresolved.append(value)
        normalized = {**payload, "paper_ids": self._dedupe_ids(resolved)}
        if unresolved:
            normalized["unresolved_paper_refs"] = unresolved
        if not normalized["paper_ids"]:
            if intent_resolved_paper_ids:
                normalized["paper_ids"] = self._dedupe_ids(intent_resolved_paper_ids)
                normalized["paper_scope_source"] = "user_intent_resolver"
        return normalized

    def _normalize_supervisor_route_payload(
        self,
        *,
        action_name: ResearchSupervisorActionName,
        payload: dict[str, Any],
        state: ResearchSupervisorState,
    ) -> dict[str, Any]:
        if action_name != "answer_question":
            return payload
        normalized = dict(payload)
        normalized["routing_authority"] = "supervisor_llm"
        qa_route = str(normalized.get("qa_route") or "").strip()
        if qa_route not in {"collection_qa", "document_drilldown", "chart_drilldown"}:
            qa_route = self._infer_answer_question_route(state=state, payload=normalized)
        normalized["qa_route"] = qa_route
        return normalized

    def _infer_answer_question_route(
        self,
        *,
        state: ResearchSupervisorState,
        payload: dict[str, Any],
    ) -> str:
        intent_name = str(state.user_intent.get("intent") or "").strip()
        has_visual_payload = any(
            str(payload.get(key) or "").strip()
            for key in ("image_path", "chart_id", "page_id", "figure_id")
        )
        if state.has_chart_input or intent_name == "figure_qa" or has_visual_payload:
            return "chart_drilldown"
        route_text = f"{state.goal} {payload.get('question') or payload.get('goal') or ''}".lower()
        collection_markers = (
            "哪篇",
            "哪些",
            "优先",
            "值得",
            "推荐",
            "对比",
            "比较",
            "which",
            "recommend",
            "compare",
        )
        if any(marker in route_text for marker in collection_markers):
            return "collection_qa"
        document_ids = [
            str(item).strip()
            for item in (payload.get("document_ids") or [])
            if str(item).strip()
        ]
        paper_ids = [
            str(item).strip()
            for item in (payload.get("paper_ids") or [])
            if str(item).strip()
        ]
        if document_ids:
            return "document_drilldown"
        if (
            state.imported_document_count > 0
            and (
                state.route_mode == "paper_follow_up"
                or intent_name in {"single_paper_qa", "document_qa"}
                or len(paper_ids) == 1
            )
        ):
            return "document_drilldown"
        return "collection_qa"

    def _best_title_match(self, normalized_value: str, by_title: dict[str, str]) -> str | None:
        value_terms = set(normalized_value.split())
        best: tuple[int, str] | None = None
        for title, paper_id in by_title.items():
            if normalized_value in title or title in normalized_value:
                return paper_id
            title_terms = set(title.split())
            overlap = len(value_terms & title_terms)
            if overlap >= max(2, min(len(value_terms), 4)):
                if best is None or overlap > best[0]:
                    best = (overlap, paper_id)
        return best[1] if best is not None else None

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", value.lower())).strip()

    def _guardrail_decision(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
    ) -> ResearchSupervisorDecision | None:
        recent_successful_actions = [
            result.task_type
            for result in results[-8:]
            if result.status == "succeeded"
        ]
        clarification_decision = self._missing_input_clarify_decision(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
        )
        if clarification_decision is not None:
            return clarification_decision
        reroute_decision = self._general_answer_reroute_decision(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
        )
        if reroute_decision is not None:
            return reroute_decision
        general_answer_done = self._general_answer_completion_decision(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
        )
        if general_answer_done is not None:
            return general_answer_done
        suggested_action_decision = self._suggested_next_action_decision(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
            recent_successful_actions=recent_successful_actions,
        )
        if suggested_action_decision is not None:
            return suggested_action_decision
        return self._evidence_gap_search_decision(
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
            recent_successful_actions=recent_successful_actions,
        )

    def _missing_input_clarify_decision(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
    ) -> ResearchSupervisorDecision | None:
        if not results:
            return None
        latest = results[-1]
        observation = self._result_observation(latest)
        missing_inputs = list(observation.get("missing_inputs") or [])
        if latest.status not in {"skipped", "failed"} or not missing_inputs:
            return None
        missing_set = {str(item).strip() for item in missing_inputs if str(item).strip()}
        if "task" in missing_set and not state.has_task:
            clarification = "你是想继续当前研究线程，还是要开始一个新的文献调研主题？"
        elif "paper_scope" in missing_set and state.has_task:
            clarification = "你想问哪篇论文或哪组论文？可以给我标题、序号，或先选中论文。"
        elif "document_file_path" in missing_set:
            clarification = "请先提供要解析的文档文件，再继续这个文档问题。"
        elif "chart_image_path" in missing_set:
            clarification = "请先提供图表图片或图表标识，再继续这个图表问题。"
        else:
            return None
        return self._guardrail_clarify(
            state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count + 1,
            clarification=clarification,
        )

    def _suggested_next_action_decision(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        recent_successful_actions: list[str],
    ) -> ResearchSupervisorDecision | None:
        if not results:
            return None
        latest = results[-1]
        observation = self._result_observation(latest)
        if latest.status == "failed":
            return None
        if observation.get("progress_made") is False and latest.status == "succeeded":
            return None
        suggestions = [
            str(item).strip()
            for item in observation.get("suggested_next_actions", [])
            if str(item).strip()
        ]
        if not suggestions:
            return None
        supported_actions = {
            "clarify_request",
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
        }
        for suggestion in suggestions:
            if suggestion not in supported_actions:
                continue
            if suggestion in recent_successful_actions:
                continue
            if suggestion == "clarify_request":
                return self._guardrail_clarify(
                    state,
                    all_messages=all_messages,
                    results=results,
                    planner_runs=planner_runs,
                    replan_count=replan_count + 1,
                    clarification="继续前我还需要你补充一下当前问题的具体对象或范围。",
                )
            if suggestion == "search_literature" and latest.task_type == "search_literature":
                continue
            payload_overrides = {
                "trigger": "observation_envelope_suggestion",
                "suggested_by": latest.task_type,
            }
            state_delta = observation.get("state_delta")
            if suggestion == "answer_question" and isinstance(state_delta, dict):
                recovery_route = str(state_delta.get("preferred_qa_route") or "").strip()
                if recovery_route in {"collection_qa", "document_drilldown", "chart_drilldown"}:
                    payload_overrides["qa_route"] = recovery_route
                    payload_overrides["qa_route_source"] = "worker_observation"
                    recovery_reason = state_delta.get("qa_recovery_reason")
                    if recovery_reason:
                        payload_overrides["qa_recovery_reason"] = recovery_reason
            return self._guardrail_worker_action(
                action_name=suggestion,  # type: ignore[arg-type]
                state=state,
                all_messages=all_messages,
                results=results,
                planner_runs=planner_runs,
                replan_count=replan_count + 1,
                thought="The latest worker result included a structured suggested-next-action hint, so the manager can continue with a lower-ambiguity follow-up step.",
                rationale=f"The latest worker reported suggested_next_actions={suggestions[:3]}, which provides a grounded follow-up action instead of forcing the manager to guess the next move.",
                phase="act",
                estimated_gain=0.72,
                estimated_cost=0.18,
                payload_overrides=payload_overrides,
                priority="high",
            )
        return None

    def _general_answer_reroute_decision(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
    ) -> ResearchSupervisorDecision | None:
        if not results:
            return None
        latest = results[-1]
        if latest.task_type != "general_answer" or latest.status != "skipped":
            return None
        payload = dict(latest.payload or {})
        if payload.get("reason") != "route_mismatch":
            return None
        suggested_action = str(payload.get("suggested_action") or "").strip()
        if suggested_action not in {"search_literature", "answer_question", "understand_document", "supervisor_understand_chart", "analyze_paper_figures"}:
            suggested_action = "answer_question" if state.has_task else "search_literature"
        return self._guardrail_worker_action(
            action_name=suggested_action,  # type: ignore[arg-type]
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count + 1,
            thought="The lightweight general-answer branch detected that this question likely belongs to a research/document/chart workflow.",
            rationale="The previous general answer step reported a route mismatch, so the supervisor should automatically reroute to the more grounded worker.",
            phase="act",
            estimated_gain=0.82,
            estimated_cost=0.28,
            payload_overrides={
                "trigger": "general_answer_route_mismatch",
                "reroute_from": "general_answer",
            },
            priority="high",
        )

    def _general_answer_completion_decision(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
    ) -> ResearchSupervisorDecision | None:
        if not results:
            return None
        latest = results[-1]
        if latest.task_type != "general_answer" or latest.status == "skipped":
            return None
        payload = dict(latest.payload or {})
        answer_type = str(payload.get("answer_type") or "").strip()
        if latest.status == "failed" or answer_type in {"provider_timeout", "provider_error"}:
            stop_reason = "通用回答模型暂时不可用，请稍后重试。"
            thought = "The general-answer provider did not produce a normal answer, so the manager should stop this chat turn without replanning it as research."
            rationale = "Provider unavailability in a general chat branch is not evidence that the user has a broad research goal."
        else:
            stop_reason = "General answer completed."
            thought = "The general-answer worker already handled the user request."
            rationale = "A completed general chat turn should not call the manager LLM again or branch into research tools."
        return self._decision(
            action_name="finalize",
            thought=thought,
            rationale=rationale,
            phase="commit",
            estimated_gain=0.0,
            estimated_cost=0.0,
            stop_reason=stop_reason,
            metadata={
                "decision_source": "manager_guardrail",
                "worker_agent": "ResearchSupervisorAgent",
                "state_update": {
                    "pending_agent_messages": [],
                    "agent_messages": all_messages,
                    "agent_results": results,
                    "completed_agent_task_ids": [
                        result.task_id for result in results if result.status in {"succeeded", "skipped"}
                    ],
                    "failed_agent_task_ids": [result.task_id for result in results if result.status == "failed"],
                    "replanned_failure_task_ids": [],
                    "planner_runs": planner_runs,
                    "replan_count": replan_count,
                    "clarification_request": None,
                    "active_plan_id": None,
                },
            },
        )

    def _evidence_gap_search_decision(
        self,
        *,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        recent_successful_actions: list[str],
    ) -> ResearchSupervisorDecision | None:
        if "search_literature" in recent_successful_actions:
            return None
        if not results:
            return None
        latest = results[-1]
        payload = dict(latest.payload or {})
        should_search = False
        reason = ""
        if latest.task_type == "answer_question" and latest.status == "succeeded":
            evidence_count = int(payload.get("evidence_count") or 0)
            confidence = payload.get("confidence")
            confidence_value = float(confidence) if isinstance(confidence, int | float) else None
            if evidence_count < 2 or (confidence_value is not None and confidence_value < 0.45):
                should_search = True
                reason = (
                    f"Local RAG evidence was thin for the question "
                    f"(evidence={evidence_count}, confidence={confidence_value if confidence_value is not None else 'empty'})."
                )
        elif latest.task_type == "analyze_papers" and latest.status in {"skipped", "failed"}:
            if payload.get("reason") in {"no_papers", "missing_task"}:
                should_search = True
                reason = "The requested paper analysis had no local candidate papers to analyze."
        if not should_search:
            return None
        target_paper_ids = self._result_paper_ids(latest)
        if target_paper_ids and not state.import_attempted:
            return self._guardrail_worker_action(
                action_name="import_papers",
                state=state,
                all_messages=all_messages,
                results=results,
                planner_runs=planner_runs,
                replan_count=replan_count,
                thought="Local RAG was insufficient, but the target paper is already in the candidate pool, so import it before searching for new papers.",
                rationale=f"{reason} The next best step is to fetch and index the targeted candidate paper for grounded follow-up QA.",
                phase="act",
                estimated_gain=0.9,
                estimated_cost=0.5,
                payload_overrides={
                    "paper_ids": target_paper_ids,
                    "trigger": "local_rag_evidence_gap",
                    "evidence_gap_query": state.goal,
                },
                priority="high",
            )
        return self._guardrail_worker_action(
            action_name="search_literature",
            state=state,
            all_messages=all_messages,
            results=results,
            planner_runs=planner_runs,
            replan_count=replan_count,
            thought="Local evidence was insufficient and no target candidate paper could be grounded locally, so broaden evidence through external literature search.",
            rationale=f"{reason} The next best step is to query external scholarly sources for additional answer clues.",
            phase="plan",
            estimated_gain=0.78,
            estimated_cost=0.42,
            payload_overrides={
                "evidence_gap_query": state.goal,
                "trigger": "local_rag_evidence_gap",
                **({"paper_ids": target_paper_ids} if target_paper_ids else {}),
            },
            priority="high",
        )

    def _result_paper_ids(self, result: AgentResultMessage) -> list[str]:
        payload = dict(result.payload or {})
        values = payload.get("paper_ids") or payload.get("analyzed_paper_ids") or []
        if not isinstance(values, list):
            return []
        return self._dedupe_ids([str(item).strip() for item in values if str(item).strip()])

    def _guardrail_finalize(
        self,
        state: ResearchSupervisorState,
        *,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        reason: str,
    ) -> ResearchSupervisorDecision:
        return self._decision(
            action_name="finalize",
            thought="Stop the manager loop because the requested work has reached a useful stopping point.",
            rationale=reason,
            phase="commit",
            estimated_gain=0.0,
            estimated_cost=0.0,
            stop_reason=reason or self._stop_reason(state),
            metadata={
                "decision_source": "manager_guardrail",
                "worker_agent": "ResearchSupervisorAgent",
                "state_update": {
                    "pending_agent_messages": [],
                    "agent_messages": all_messages,
                    "agent_results": results,
                    "completed_agent_task_ids": [result.task_id for result in results if result.status in {"succeeded", "skipped"}],
                    "failed_agent_task_ids": [result.task_id for result in results if result.status == "failed"],
                    "replanned_failure_task_ids": [],
                    "planner_runs": planner_runs,
                    "replan_count": replan_count,
                    "clarification_request": None,
                    "active_plan_id": None,
                },
            },
        )

    def _guardrail_clarify(
        self,
        state: ResearchSupervisorState,
        *,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        clarification: str,
    ) -> ResearchSupervisorDecision:
        return self._decision(
            action_name="clarify_request",
            thought="The manager needs one focused clarification before any worker action would be high signal.",
            rationale="The current request is underspecified enough that asking a targeted clarification is more reliable than guessing the route.",
            phase="commit",
            estimated_gain=0.32,
            estimated_cost=0.02,
            action_input={
                "clarification_question": clarification,
                "route_mode": state.route_mode,
            },
            stop_reason=clarification,
            metadata={
                "decision_source": "manager_guardrail",
                "worker_agent": "ResearchSupervisorAgent",
                "clarification_request": clarification,
                "state_update": {
                    "pending_agent_messages": [],
                    "agent_messages": all_messages,
                    "agent_results": results,
                    "completed_agent_task_ids": [result.task_id for result in results if result.status in {"succeeded", "skipped"}],
                    "failed_agent_task_ids": [result.task_id for result in results if result.status == "failed"],
                    "replanned_failure_task_ids": [],
                    "planner_runs": planner_runs,
                    "replan_count": replan_count,
                    "clarification_request": clarification,
                    "active_plan_id": None,
                },
            },
        )

    def _guardrail_worker_action(
        self,
        *,
        action_name: ResearchSupervisorActionName,
        state: ResearchSupervisorState,
        all_messages: list[AgentMessage],
        results: list[AgentResultMessage],
        planner_runs: int,
        replan_count: int,
        thought: str,
        rationale: str,
        phase: Literal["plan", "act", "reflect", "commit"],
        estimated_gain: float,
        estimated_cost: float,
        payload_overrides: dict[str, Any] | None = None,
        priority: Literal["low", "medium", "high", "critical"] = "medium",
    ) -> ResearchSupervisorDecision:
        worker_agent = self._worker_for_action(action_name, None)
        plan_id = f"guardrail_plan_{uuid4().hex[:12]}"
        payload = {
            **self._default_payload_for_action(action_name, state),
            **(payload_overrides or {}),
        }
        payload = self._normalize_payload_paper_scope(
            action_name=action_name,
            payload=payload,
            state=state,
        )
        payload = self._normalize_supervisor_route_payload(
            action_name=action_name,
            payload=payload,
            state=state,
        )
        instruction = self._default_instruction_for_action(action_name, state, payload)
        active_message = AgentMessage(
            task_id=f"guardrail_task_{uuid4().hex[:12]}",
            agent_from="ResearchSupervisorAgent",
            agent_to=worker_agent,
            task_type=self._task_type_for_action(action_name),
            instruction=instruction,
            payload=payload,
            context_slice={},
            priority=priority,
            expected_output_schema=self._expected_output_schema_for_action(action_name),
            metadata={
                "plan_id": plan_id,
                "decision_source": "manager_guardrail",
                "manager_action": action_name,
            },
        )
        agent_messages = [*all_messages, active_message]
        return self._decision(
            action_name=action_name,
            thought=thought,
            rationale=rationale,
            phase=phase,
            estimated_gain=estimated_gain,
            estimated_cost=estimated_cost,
            action_input={**payload, "instruction": instruction},
            metadata={
                "decision_source": "manager_guardrail",
                "worker_agent": worker_agent,
                "worker_task_type": self._task_type_for_action(action_name),
                "plan_id": plan_id,
                "active_message": active_message,
                "state_update": {
                    "pending_agent_messages": [active_message],
                    "agent_messages": agent_messages,
                    "agent_results": results,
                    "completed_agent_task_ids": [result.task_id for result in results if result.status in {"succeeded", "skipped"}],
                    "failed_agent_task_ids": [result.task_id for result in results if result.status == "failed"],
                    "replanned_failure_task_ids": [],
                    "planner_runs": planner_runs + 1,
                    "replan_count": replan_count,
                    "clarification_request": None,
                    "active_plan_id": plan_id,
                    "new_topic_detected": state.new_topic_detected,
                },
            },
        )

    def _available_actions(self, state: ResearchSupervisorState) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        if state.user_intent.get("needs_clarification"):
            actions.append(self._action_descriptor("clarify_request", "ResearchSupervisorAgent", "Ask a focused clarification question when the user's target or topic boundary is ambiguous.", state=state))
        actions.append(self._action_descriptor("general_answer", "GeneralAnswerAgent", "Answer a general non-research question directly when the user does not need literature search, local RAG, document parsing, or chart reasoning.", state=state))
        if state.preference_recommendation_requested or state.known_interest_count > 0:
            actions.append(
                self._action_descriptor(
                    "recommend_from_preferences",
                    "PreferenceMemoryAgent",
                    "Recommend recent papers from the user's long-term interest profile when the user asks for broad worth-reading suggestions.",
                    state=state,
                )
            )
        if state.has_document_input and not state.document_understood:
            actions.append(self._action_descriptor("understand_document", "ResearchDocumentAgent", "Parse and ground the uploaded document before other research actions.", state=state))
        intent_name = str(state.user_intent.get("intent") or "").strip()
        if (state.has_chart_input and not state.chart_understood) or intent_name == "figure_qa":
            actions.append(self._action_descriptor("supervisor_understand_chart", "ChartAnalysisAgent", "Understand or analyze a chart, figure, or diagram from an uploaded image.", state=state))
        if not state.has_task:
            actions.append(self._action_descriptor("search_literature", "LiteratureScoutAgent", "Start a new research exploration and create the initial task, candidate papers, and workspace.", state=state))
        else:
            task_actions = [
                self._action_descriptor("search_literature", "LiteratureScoutAgent", "Refresh or expand the literature search when the current candidate set is weak or stale.", state=state),
                self._action_descriptor("write_review", "ResearchWriterAgent", "Synthesize the current research workspace into a grounded review or progress report.", state=state),
                self._action_descriptor("import_papers", "ResearchKnowledgeAgent", "Import promising papers into the local research workspace when the user wants grounded QA, evidence retrieval, or local ingestion.", state=state),
                self._action_descriptor("sync_to_zotero", "ResearchKnowledgeAgent", "Sync targeted candidate papers to Zotero when the user asks to import, save, or add papers into their Zotero library.", state=state),
                self._action_descriptor("answer_question", "ResearchQAAgent", "Answer a collection, document, or chart question using the selected QA route and current workspace.", state=state),
                self._action_descriptor("analyze_papers", "PaperAnalysisAgent", "Analyze the selected papers and answer comparison, explanation, or recommendation questions.", state=state),
                self._action_descriptor("compress_context", "ResearchKnowledgeAgent", "Compress a large workspace into reusable summaries before deeper reasoning.", state=state),
            ]
            if state.imported_document_count > 0:
                task_actions.append(self._action_descriptor("analyze_paper_figures", "ChartAnalysisAgent", "Extract and analyze figures, charts, or diagrams from an imported paper's PDF. Use when the user asks about a figure, system diagram, architecture, experimental result plot, or any visual element in a paper.", state=state))
            actions.extend(task_actions)
        actions.append(self._action_descriptor("finalize", "ResearchSupervisorAgent", "Stop when additional tool use is low value or clarification from the user is needed.", state=state))
        return sorted(actions, key=lambda item: item.get("priority_score", 0.0), reverse=True)

    def _action_descriptor(
        self,
        action_name: str,
        worker_agent: str,
        when_to_use: str,
        *,
        state: ResearchSupervisorState,
    ) -> dict[str, Any]:
        return {
            "action_name": action_name,
            "worker_agent": worker_agent,
            "when_to_use": when_to_use,
            "default_task_type": self._task_type_for_action(action_name),  # type: ignore[arg-type]
            "expected_output_schema": self._expected_output_schema_for_action(action_name),  # type: ignore[arg-type]
            "priority_score": self._action_priority_score(action_name, state),
            "visibility_reason": self._action_visibility_reason(action_name, state),
        }

    def _action_priority_score(self, action_name: str, state: ResearchSupervisorState) -> float:
        score = 0.1
        if action_name in state.latest_suggested_next_actions:
            score += 0.42
        if action_name == "general_answer" and state.route_mode == "general_chat":
            score += 1.0
        if action_name == "clarify_request" and state.user_intent.get("needs_clarification"):
            score += 1.05
        if action_name == "clarify_request" and state.latest_missing_inputs:
            score += 0.4
        if action_name == "search_literature" and (not state.has_task or state.new_topic_detected or state.route_mode == "research_discovery"):
            score += 0.9
        if action_name == "search_literature" and "papers" in state.latest_missing_inputs:
            score += 0.25
        if action_name == "answer_question" and state.route_mode in {"research_follow_up", "paper_follow_up"}:
            score += 0.78
        if action_name == "answer_question" and state.latest_result_task_type == "search_literature" and state.latest_progress_made:
            score += 0.18
        if action_name == "recommend_from_preferences" and state.preference_recommendation_requested:
            score += 1.02
        if action_name == "recommend_from_preferences" and state.known_interest_count > 0:
            score += 0.28
        if action_name == "analyze_papers" and state.paper_analysis_requested:
            score += 0.74
        if action_name == "import_papers" and state.has_import_candidates and not state.import_attempted:
            score += 0.68
        if action_name == "import_papers" and state.latest_result_task_type == "search_literature" and state.latest_progress_made:
            score += 0.15
        if action_name == "compress_context" and state.context_compression_needed:
            score += 0.45
        if action_name == "finalize" and state.user_intent.get("needs_clarification"):
            score += 0.85
        if action_name == "finalize" and state.latest_progress_made is False and not state.latest_suggested_next_actions:
            score += 0.18
        intent_name = str(state.user_intent.get("intent") or "").strip()
        if action_name == "analyze_paper_figures" and intent_name == "figure_qa" and state.imported_document_count > 0:
            score += 0.88
        if action_name == "supervisor_understand_chart" and state.has_chart_input and not state.chart_understood:
            score += 0.92
        return round(score, 3)

    def _action_visibility_reason(self, action_name: str, state: ResearchSupervisorState) -> str:
        if action_name == "general_answer" and state.route_mode == "general_chat":
            return "The current turn looks like general chat and should not inherit the active research scope."
        if action_name == "search_literature" and state.new_topic_detected:
            return "A new topic appears to have started, so discovery should rebuild scope instead of inheriting the previous paper focus."
        if action_name == "recommend_from_preferences" and state.preference_recommendation_requested:
            return "The user is asking for broad reading recommendations, so the preference specialist can reuse long-term interests instead of the current paper scope."
        if action_name == "analyze_paper_figures" and str(state.user_intent.get("intent") or "").strip() == "figure_qa":
            return "The user is asking about a figure, diagram, or chart in an imported paper. Use analyze_paper_figures to extract and analyze the visual element directly from the PDF."
        if action_name == "answer_question" and state.route_mode == "paper_follow_up":
            return "The user appears to be following up on the current paper or imported collection."
        if action_name in state.latest_suggested_next_actions:
            return "This action was explicitly suggested by the latest worker observation and is therefore a lower-ambiguity next step."
        if action_name == "finalize" and state.user_intent.get("needs_clarification"):
            return "The request still needs clarification before a high-signal worker step is safe."
        if action_name == "clarify_request":
            return "Visible because the current user turn is ambiguous enough that a clarification question is safer than guessing the next route."
        return "Visible under the current workspace state and available for supervisor selection."

    def _llm_prompt(self) -> str:
        return (
            "You are the autonomous research manager for a multi-agent literature research system.\n"
            "Choose exactly one next action. Prefer worker autonomy over fixed pipelines.\n"
            "Use the available actions, current workspace state, recent task outcomes, and evidence gaps.\n"
            "You will receive guardrail_hints from the rule-based pre-screening layer. "
            "These are keyword-based suggestions that may be semantically incorrect. "
            "Always interpret the user's message holistically and override any guardrail hint "
            "when the true semantic intent of the user's request disagrees with the suggested action.\n"
            "Use state.user_intent as a hint, not as a hard rule. "
            "If it says needs_clarification, evaluate whether clarification is truly needed "
            "based on the full context — for example, if there is exactly 1 imported document and the user says '这篇论文', "
            "it almost certainly refers to that document rather than requiring clarification; "
            "if the user says '讲解导入论文的方法', the word '导入' is part of the noun phrase meaning 'the imported paper', "
            "not an import command.\n"
            "Use state.route_mode, state.active_thread_topic, state.topic_continuity_score, and state.new_topic_detected to decide whether to continue the current research thread or start a fresh discovery path.\n"
            "Use state.latest_result_task_type, state.latest_progress_made, state.latest_result_confidence, state.latest_missing_inputs, and state.latest_suggested_next_actions as strong feedback from the most recent worker execution.\n"
            "When state.route_mode is general_chat or state.should_ignore_research_context is true, avoid inheriting the previous paper scope unless the user explicitly re-enters it.\n"
            "When the user asks about a figure, diagram, chart, architecture, or system block diagram in an imported paper, choose analyze_paper_figures to extract and analyze the visual element directly from the PDF.\n"
            "When the user asks a general question that does not require literature search, paper import, local evidence retrieval, document parsing, or chart analysis, choose general_answer.\n"
            "When the user asks for broad, unscoped papers worth reading and the request should use the user's long-term interests instead of the current paper scope, choose recommend_from_preferences.\n"
            "For simple, single-intent requests, decide the single best next worker action now.\n"
            "For complex, multi-intent requests (e.g. 'search papers, compare them, then import the best'), "
            "produce a multi-step plan in the 'plan' field. Each step should have step_id, action, instruction, "
            "params, and depends_on (list of step_ids this step depends on). Set action_name to the first step's action. "
            "The runtime will execute steps in order, advancing automatically after each succeeds. "
            "If a step fails, the plan is cleared and you will be called again to replan. "
            "Only use plan for genuinely multi-step requests; do not plan for single actions.\n"
            "Use finalize when the workspace is already useful, a requested action is complete, a clarification is required, or no action has enough marginal value.\n"
            "Do not repeat a search, answer, paper analysis, import, or writing action that already succeeded in the recent results unless the state clearly changed.\n"
            "When selecting a worker action, provide a concrete instruction that helps the worker act independently.\n"
            "Interpret user intent semantically rather than mechanically: if the user asks to import/save/add a paper into Zotero or a citation manager, choose sync_to_zotero; "
            "if the user asks to ingest/import papers for grounded QA, local evidence retrieval, or workspace use, choose import_papers.\n"
            "When the user refers to candidate papers by ordinal, title, topic, or phrases like 'this paper'/'these papers', "
            "resolve the intended candidate papers from state.candidate_papers and include their exact paper_ids in payload.paper_ids. "
            "If the user asks a follow-up that omits the paper subject and state.active_paper_ids is non-empty, "
            "treat state.active_paper_ids as the default paper scope unless the user names a different paper. "
            "Use payload.paper_ids for answer_question, analyze_papers, import_papers, sync_to_zotero, and compress_context when the user targets a subset; "
            "leave it empty only when the user clearly asks about the whole collection or a new search.\n"
            "For answer_question, you must also decide payload.qa_route as one of collection_qa, document_drilldown, or chart_drilldown. "
            "Downstream services should execute your route instead of deciding it again.\n"
            "Keep payload fields compact and directly actionable.\n"
            "Before choosing an action, first resolve the user's true intent from their message. "
            "The heuristic intent in state.user_intent is a hint but may be inaccurate — "
            "override it with resolved_intent if you disagree. "
            "If the user refers to papers by ordinal, title, or phrases like '这篇'/'第一篇'/'p1', "
            "resolve the actual paper_ids from state.candidate_papers into resolved_paper_ids."
        )

    def _state_snapshot(self, state: ResearchSupervisorState) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "goal": state.goal,
            "mode": state.mode,
            "route_mode": state.route_mode,
            "has_task": state.has_task,
            "has_report": state.has_report,
            "paper_count": state.paper_count,
            "imported_document_count": state.imported_document_count,
            "has_document_input": state.has_document_input,
            "has_chart_input": state.has_chart_input,
            "workspace_stage": state.workspace_stage,
            "last_action_name": state.last_action_name,
            "latest_result_task_type": state.latest_result_task_type,
            "latest_result_status": state.latest_result_status,
            "user_intent": dict(state.user_intent),
        }
        _conditional = {
            "active_thread_topic": state.active_thread_topic,
            "topic_continuity_score": state.topic_continuity_score,
            "new_topic_detected": state.new_topic_detected,
            "should_ignore_research_context": state.should_ignore_research_context,
            "active_paper_ids": list(state.active_paper_ids),
            "import_attempted": state.import_attempted,
            "answer_attempted": state.answer_attempted,
            "context_compression_needed": state.context_compression_needed,
            "context_compressed": state.context_compressed,
            "paper_analysis_completed": state.paper_analysis_completed,
            "paper_analysis_requested": state.paper_analysis_requested,
            "preference_recommendation_requested": state.preference_recommendation_requested,
            "analysis_focus": state.analysis_focus,
            "failed_actions": list(state.failed_actions),
            "latest_progress_made": state.latest_progress_made,
            "latest_result_confidence": state.latest_result_confidence,
            "latest_missing_inputs": list(state.latest_missing_inputs),
            "latest_suggested_next_actions": list(state.latest_suggested_next_actions),
        }
        _defaults: dict[str, Any] = {
            "active_thread_topic": None,
            "topic_continuity_score": None,
            "new_topic_detected": False,
            "should_ignore_research_context": False,
            "active_paper_ids": [],
            "import_attempted": False,
            "answer_attempted": False,
            "context_compression_needed": False,
            "context_compressed": False,
            "paper_analysis_completed": False,
            "paper_analysis_requested": False,
            "preference_recommendation_requested": False,
            "analysis_focus": None,
            "failed_actions": [],
            "latest_progress_made": None,
            "latest_result_confidence": None,
            "latest_missing_inputs": [],
            "latest_suggested_next_actions": [],
        }
        for key, value in _conditional.items():
            if value != _defaults.get(key):
                snapshot[key] = value
        if state.candidate_papers:
            snapshot["candidate_papers"] = [
                {"index": i + 1, "paper_id": p.get("paper_id", ""), "title": p.get("title", "")}
                for i, p in enumerate(list(state.candidate_papers)[:10])
            ]
        if state.execution_plan:
            snapshot["execution_plan"] = state.execution_plan
        return snapshot

    def _message_snapshot(self, message: AgentMessage) -> dict[str, Any]:
        return {
            "task_id": message.task_id,
            "agent_to": message.agent_to,
            "task_type": message.task_type,
            "instruction": message.instruction,
            "payload": dict(message.payload),
            "priority": message.priority,
            "depends_on": list(message.depends_on),
            "metadata": dict(message.metadata),
        }

    def _result_snapshot(self, result: AgentResultMessage) -> dict[str, Any]:
        return {
            "task_id": result.task_id,
            "agent_from": result.agent_from,
            "task_type": result.task_type,
            "status": result.status,
            "instruction": result.instruction,
            "payload": dict(result.payload),
            "observation_envelope": self._result_observation(result),
            "evaluation": result.evaluation.model_dump(mode="json") if result.evaluation is not None else None,
            "metadata": dict(result.metadata),
        }

    def _result_observation(self, result: AgentResultMessage) -> dict[str, Any]:
        payload = dict(result.payload or {})
        observation = payload.get("observation_envelope")
        if isinstance(observation, dict):
            return dict(observation)
        return {}

    def _state_with_recent_result_signal(
        self,
        state: ResearchSupervisorState,
        *,
        results: list[AgentResultMessage],
    ) -> ResearchSupervisorState:
        if not results:
            return state
        latest = results[-1]
        observation = self._result_observation(latest)
        confidence = observation.get("confidence")
        return replace(
            state,
            latest_result_task_type=latest.task_type,
            latest_result_status=latest.status,
            latest_progress_made=(
                bool(observation.get("progress_made"))
                if observation.get("progress_made") is not None
                else None
            ),
            latest_result_confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            latest_missing_inputs=[
                str(item).strip()
                for item in observation.get("missing_inputs", [])
                if str(item).strip()
            ],
            latest_suggested_next_actions=[
                str(item).strip()
                for item in observation.get("suggested_next_actions", [])
                if str(item).strip()
            ],
        )

    def _normalize_action_name(self, action_name: str) -> ResearchSupervisorActionName:
        normalized = action_name.strip()
        allowed = {
            "clarify_request",
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
            "finalize",
        }
        if normalized in {"compare_papers", "recommend_papers"}:
            normalized = "analyze_papers"
        if normalized in {"recommend_recent_papers", "personalized_recommendation"}:
            normalized = "recommend_from_preferences"
        if normalized not in allowed:
            raise ValueError(f"unsupported LLM manager action: {action_name}")
        return normalized  # type: ignore[return-value]

    def _worker_for_action(self, action_name: ResearchSupervisorActionName, worker_agent: str | None) -> str:
        defaults = {
            "clarify_request": "ResearchSupervisorAgent",
            "search_literature": "LiteratureScoutAgent",
            "write_review": "ResearchWriterAgent",
            "import_papers": "ResearchKnowledgeAgent",
            "sync_to_zotero": "ResearchKnowledgeAgent",
            "answer_question": "ResearchQAAgent",
            "general_answer": "GeneralAnswerAgent",
            "recommend_from_preferences": "PreferenceMemoryAgent",
            "analyze_papers": "PaperAnalysisAgent",
            "compress_context": "ResearchKnowledgeAgent",
            "understand_document": "ResearchDocumentAgent",
            "supervisor_understand_chart": "ChartAnalysisAgent",
            "analyze_paper_figures": "ChartAnalysisAgent",
            "finalize": "ResearchSupervisorAgent",
        }
        if action_name == "answer_question":
            return "ResearchQAAgent"
        if worker_agent and worker_agent.strip():
            return worker_agent.strip()
        return defaults[action_name]

    def _task_type_for_action(self, action_name: ResearchSupervisorActionName) -> str:
        mapping = {
            "clarify_request": "clarify_request",
            "search_literature": "search_literature",
            "write_review": "write_review",
            "import_papers": "import_papers",
            "sync_to_zotero": "sync_to_zotero",
            "answer_question": "answer_question",
            "general_answer": "general_answer",
            "recommend_from_preferences": "recommend_from_preferences",
            "analyze_papers": "analyze_papers",
            "compress_context": "compress_context",
            "understand_document": "understand_document",
            "supervisor_understand_chart": "supervisor_understand_chart",
            "analyze_paper_figures": "analyze_paper_figures",
            "finalize": "finalize",
        }
        return mapping[action_name]

    def _default_payload_for_action(
        self,
        action_name: ResearchSupervisorActionName,
        state: ResearchSupervisorState,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"goal": state.goal, "mode": state.mode}
        if action_name == "answer_question":
            payload["routing_authority"] = "supervisor_llm"
        if action_name == "clarify_request":
            payload["clarification_kind"] = "goal_scope"
        if action_name == "general_answer":
            payload["ignore_research_context"] = state.should_ignore_research_context or state.route_mode == "general_chat"
        if action_name == "recommend_from_preferences":
            payload.update(
                {
                    "top_k": state.recommendation_top_k or 6,
                    "days_back": 30,
                    "use_long_term_profile": True,
                }
            )
        if action_name == "analyze_papers":
            if state.analysis_focus:
                payload["analysis_focus"] = state.analysis_focus
            payload["dimensions"] = list(state.comparison_dimensions)
            payload["top_k"] = state.recommendation_top_k
            if state.recommendation_goal:
                payload["recommendation_goal"] = state.recommendation_goal
        if action_name == "import_papers":
            payload.update(
                {
                    "auto_import": state.auto_import,
                    "import_top_k": state.import_top_k,
                    "selected_paper_count": state.selected_paper_count,
                    "importable_paper_count": state.importable_paper_count,
                }
            )
        if action_name == "sync_to_zotero":
            payload.update({"collection_name": None})
        return payload

    def _default_instruction_for_action(
        self,
        action_name: ResearchSupervisorActionName,
        state: ResearchSupervisorState,
        payload: dict[str, Any],
    ) -> str:
        if action_name == "search_literature":
            return f"Search and curate literature for '{state.goal}', then update the research workspace with the strongest candidate papers."
        if action_name == "clarify_request":
            return f"Ask the user a focused clarification question before continuing work on '{state.goal}'."
        if action_name == "write_review":
            return f"Write or refine a grounded literature review for '{state.goal}' using the current workspace and citations."
        if action_name == "import_papers":
            if payload.get("paper_ids"):
                return f"Import and index the targeted candidate papers for '{state.goal}' so follow-up QA can use local evidence."
            return f"Import the most valuable PDF-backed papers for '{state.goal}' into the evidence base."
        if action_name == "sync_to_zotero":
            if payload.get("paper_ids"):
                return f"Sync the targeted candidate papers for '{state.goal}' into Zotero, reusing existing items when possible."
            return f"Sync the most relevant candidate papers for '{state.goal}' into Zotero for citation management."
        if action_name == "answer_question":
            return f"Answer the user's research question '{state.goal}' using imported evidence and the current workspace."
        if action_name == "general_answer":
            return f"Answer the user's general question '{state.goal}' directly without using the research pipeline unless the question clearly requires it."
        if action_name == "recommend_from_preferences":
            return (
                f"Use the user's long-term preference memory to recommend recent papers for '{state.goal}', "
                "covering the strongest recurring topics and giving concise reasons for each recommendation."
            )
        if action_name == "analyze_papers":
            dimensions = ", ".join(str(item) for item in payload.get("dimensions") or [])
            focus = str(payload.get("analysis_focus") or "").strip()
            focus_hint = f" Focus on {dimensions}." if dimensions else ""
            if focus == "recommend":
                return f"Analyze the selected papers for '{state.goal}' and recommend the most worthwhile papers to read next with concise reasons."
            if focus == "explain":
                return f"Analyze the selected papers for '{state.goal}' and explain each paper's contribution, method, and boundary clearly."
            return f"Analyze the selected papers for '{state.goal}' and surface the most useful differences, strengths, and next-step reading advice.{focus_hint}"
        if action_name == "compress_context":
            return f"Compress the current research context for '{state.goal}' so later workers can reason over a denser summary."
        if action_name == "understand_document":
            return "Parse and ground the uploaded document into reusable research evidence."
        if action_name == "supervisor_understand_chart":
            return "Understand the uploaded chart and convert it into structured evidence for later reasoning."
        if action_name == "analyze_paper_figures":
            return f"Extract and analyze figures from an imported paper's PDF to answer the user's figure question about '{state.goal}'."
        return "Stop and return control to the user."

    def _expected_output_schema_for_action(self, action_name: ResearchSupervisorActionName) -> dict[str, Any]:
        schemas: dict[str, dict[str, Any]] = {
            "clarify_request": {"clarification_question": "str", "route_mode": "str"},
            "search_literature": {"task_id": "str", "paper_count": "int", "report_id": "str|null"},
            "write_review": {"task_id": "str", "report_id": "str", "report_word_count": "int"},
            "import_papers": {"paper_ids": "list[str]", "imported_count": "int", "failed_count": "int"},
            "sync_to_zotero": {"paper_ids": "list[str]", "synced_count": "int", "failed_count": "int"},
            "answer_question": {
                "task_id": "str",
                "document_ids": "list[str]",
                "evidence_count": "int",
                "qa_route": "collection_qa|document_drilldown|chart_drilldown",
            },
            "general_answer": {"answer": "str", "confidence": "float", "key_points": "list[str]"},
            "recommend_from_preferences": {
                "recommendations": "list[RecommendedPaper]",
                "topics_used": "list[str]",
                "days_back": "int",
            },
            "analyze_papers": {"answer": "str", "focus": "str", "recommended_paper_ids": "list[str]"},
            "compress_context": {"paper_count": "int", "summary_count": "int"},
            "understand_document": {"document_id": "str", "page_count": "int"},
            "supervisor_understand_chart": {"chart_id": "str", "chart_type": "str|null"},
            "analyze_paper_figures": {"paper_id": "str", "figure_id": "str", "answer": "str"},
            "finalize": {"stop_reason": "str"},
        }
        return schemas[action_name]

    _SERIALIZE_MAX_HISTORY_TURNS: int = 3
    _SERIALIZE_MAX_SUMMARIES: int = 5

    def _serialize_context_slice(self, context_slice: ResearchContextSlice | None) -> dict[str, Any]:
        if context_slice is None:
            return {}
        if hasattr(context_slice, "model_dump"):
            data = context_slice.model_dump(mode="json")
        elif isinstance(context_slice, dict):
            data = dict(context_slice)
        else:
            return {}
        if "session_history" in data:
            data["session_history"] = data["session_history"][-self._SERIALIZE_MAX_HISTORY_TURNS:]
        if "relevant_summaries" in data:
            data["relevant_summaries"] = data["relevant_summaries"][:self._SERIALIZE_MAX_SUMMARIES]
        return data

    def _llm_unavailable_decision(
        self,
        state: ResearchSupervisorState,
        *,
        pending_messages: list[AgentMessage] | None = None,
        agent_messages: list[AgentMessage] | None = None,
        agent_results: list[AgentResultMessage] | None = None,
        completed_task_ids: list[str] | None = None,
        failed_task_ids: list[str] | None = None,
        replanned_failure_task_ids: list[str] | None = None,
        planner_runs: int = 0,
        replan_count: int = 0,
        clarification_request: str | None = None,
        active_plan_id: str | None = None,
        reason: str | None = None,
    ) -> ResearchSupervisorDecision:
        results = self._evaluate_results(
            results=list(agent_results or []),
            agent_messages=list(agent_messages or []),
        )
        state = self._state_with_recent_result_signal(state, results=results)
        state_update = {
            "pending_agent_messages": list(pending_messages or []),
            "agent_messages": list(agent_messages or []),
            "agent_results": results,
            "completed_agent_task_ids": self._dedupe_ids(completed_task_ids or []),
            "failed_agent_task_ids": self._dedupe_ids(failed_task_ids or []),
            "replanned_failure_task_ids": self._dedupe_ids(replanned_failure_task_ids or []),
            "planner_runs": planner_runs,
            "replan_count": replan_count,
            "clarification_request": clarification_request,
            "active_plan_id": active_plan_id,
            "new_topic_detected": state.new_topic_detected,
        }
        latest_result = results[-1] if results else None
        intent_name = str(state.user_intent.get("intent") or "").strip()
        is_auto_general_chat = state.mode == "auto" and (
            state.route_mode == "general_chat"
            or intent_name == "general_answer"
        )
        latest_general_answer_result = (
            latest_result is not None
            and latest_result.task_type == "general_answer"
            and latest_result.status != "skipped"
        )
        if is_auto_general_chat or latest_general_answer_result:
            return self._decision(
                action_name="finalize",
                thought="The general-answer provider is unavailable, so the manager should stop without converting the turn into a research clarification.",
                rationale="A failed general chat turn is not evidence that the user has a broad research goal.",
                phase="commit",
                estimated_gain=0.0,
                estimated_cost=0.0,
                stop_reason="通用回答模型暂时不可用，请稍后重试。",
                metadata={
                    "decision_source": "guardrail",
                    "worker_agent": "ResearchSupervisorAgent",
                    "llm_error": reason,
                    "state_update": {
                        **state_update,
                        "pending_agent_messages": [],
                        "clarification_request": None,
                    },
                },
            )
        fallback = self._fallback_rule_decision(
            state,
            planner_runs=planner_runs,
            replan_count=replan_count,
            clarification_request=clarification_request,
            state_update=state_update,
        )
        if fallback is not None:
            return fallback
        return self._decision(
            action_name="finalize",
            thought="Manager cannot find a safe next tool step without an LLM decision channel.",
            rationale="The current workspace does not match any deterministic fallback path, so stopping avoids low-signal orchestration.",
            phase="commit",
            estimated_gain=0.0,
            estimated_cost=0.0,
            stop_reason=reason or self._stop_reason(state),
            metadata={
                "decision_source": "guardrail",
                "worker_agent": "ResearchSupervisorAgent",
                "state_update": {**state_update, "pending_agent_messages": []},
            },
        )

    def decide_next_action(
        self,
        state: ResearchSupervisorState,
        *,
        pending_messages: list[AgentMessage] | None = None,
        agent_messages: list[AgentMessage] | None = None,
        agent_results: list[AgentResultMessage] | None = None,
        completed_task_ids: list[str] | None = None,
        failed_task_ids: list[str] | None = None,
        replanned_failure_task_ids: list[str] | None = None,
        planner_runs: int = 0,
        replan_count: int = 0,
        context_slice: ResearchContextSlice | None = None,
        clarification_request: str | None = None,
        active_plan_id: str | None = None,
    ) -> ResearchSupervisorDecision:
        results = self._evaluate_results(
            results=list(agent_results or []),
            agent_messages=list(agent_messages or []),
        )
        state_update = {
            "pending_agent_messages": list(pending_messages or []),
            "agent_messages": list(agent_messages or []),
            "agent_results": results,
            "completed_agent_task_ids": self._dedupe_ids(completed_task_ids or []),
            "failed_agent_task_ids": self._dedupe_ids(failed_task_ids or []),
            "replanned_failure_task_ids": self._dedupe_ids(replanned_failure_task_ids or []),
            "planner_runs": planner_runs,
            "replan_count": replan_count,
            "clarification_request": clarification_request,
            "active_plan_id": active_plan_id,
            "new_topic_detected": state.new_topic_detected,
        }
        return self._decision(
            action_name="finalize",
            thought="Manager cannot find a safe next tool step without an LLM decision channel.",
            rationale="Rule-based planning has been removed; the manager requires LLM-driven decisions on the synchronous fallback path.",
            phase="commit",
            estimated_gain=0.0,
            estimated_cost=0.0,
            stop_reason="Rule-based planning has been removed; manager requires LLM-driven decisions.",
            metadata={
                "decision_source": "guardrail",
                "worker_agent": "ResearchSupervisorAgent",
                "state_update": {**state_update, "pending_agent_messages": []},
            },
        )

    def decompose_with_llm(
        self,
        state: ResearchSupervisorState,
        *,
        context_slice: ResearchContextSlice | None = None,
    ) -> ResearchSupervisorDecision:
        return self.decide_next_action(
            state,
            context_slice=context_slice,
        )

    def _should_import(self, state: ResearchSupervisorState) -> bool:
        if state.import_attempted or not state.has_import_candidates:
            return False
        if state.selected_paper_count > 0:
            return True
        if state.mode == "import":
            return True
        if state.mode in {"auto", "research"} and state.auto_import and state.import_top_k > 0:
            return True
        return False

    def _should_understand_document(self, state: ResearchSupervisorState) -> bool:
        if state.document_understood:
            return False
        return state.has_document_input or state.mode == "document"

    def _should_understand_chart(self, state: ResearchSupervisorState) -> bool:
        if state.chart_understood:
            return False
        return state.has_chart_input or state.mode == "chart"

    def _should_answer(self, state: ResearchSupervisorState) -> bool:
        if state.answer_attempted:
            return False
        if state.mode == "qa":
            return True
        if state.mode == "import":
            return state.import_attempted and state.imported_document_count > 0
        if state.workspace_ready and state.evidence_gap_count == 0 and state.open_todo_count > 0:
            return False
        if state.mode == "auto" and state.task_id and self._looks_like_follow_up_question(state.goal):
            return True
        return False

    def _should_recommend_from_preferences(self, state: ResearchSupervisorState) -> bool:
        if not state.preference_recommendation_requested:
            return False
        return state.known_interest_count > 0

    def _should_analyze(self, state: ResearchSupervisorState) -> bool:
        if state.paper_analysis_completed or not state.paper_analysis_requested:
            return False
        required_papers = 2 if state.analysis_focus == "compare" else 1
        return max(state.paper_count, state.selected_paper_count) >= required_papers

    def _should_compress_context(self, state: ResearchSupervisorState) -> bool:
        if state.context_compressed:
            return False
        return state.context_compression_needed

    def _context_exceeds_budget(self, context_slice: ResearchContextSlice | None, *, budget_chars: int = 120_000) -> bool:
        """Return True when the serialized context_slice is too large for a safe LLM call."""
        if context_slice is None:
            return False
        import json
        try:
            serialized = self._serialize_context_slice(context_slice)
            return len(json.dumps(serialized, ensure_ascii=False, default=str)) > budget_chars
        except Exception:
            return False

    def _truncate_context_slice(
        self,
        context_slice: ResearchContextSlice,
        *,
        budget_chars: int = 100_000,
    ) -> ResearchContextSlice:
        """Hard-truncate a context slice so it fits within *budget_chars* when serialized.

        This is the safety net for cases where compression already ran but the
        slice is still too large (e.g. huge metadata, recalled memories, or
        imported-paper metadata carrying raw page content).

        Strategy — progressively strip the heaviest fields:
        1. Clear metadata (often carries raw page content)
        2. Clear memory_context recalled memories
        3. Truncate session_history to last 3 turns
        4. Truncate imported_papers metadata
        5. Strip relevant_summaries to 3
        6. Final fallback: keep only topic, goals, selected_papers
        """
        import json

        def _size(s: ResearchContextSlice) -> int:
            try:
                return len(json.dumps(self._serialize_context_slice(s), ensure_ascii=False, default=str))
            except Exception:
                return budget_chars + 1

        if _size(context_slice) <= budget_chars:
            return context_slice

        truncated = context_slice.model_copy(deep=True)

        # Step 1: strip metadata (most common culprit)
        truncated.metadata = {"truncated": True, "context_scope": truncated.metadata.get("context_scope", "manager")}
        if _size(truncated) <= budget_chars:
            return truncated

        # Step 2: strip memory_context
        truncated.memory_context = {}
        if _size(truncated) <= budget_chars:
            return truncated

        # Step 3: cap session_history to last 3 turns
        truncated.session_history = truncated.session_history[-3:]
        if _size(truncated) <= budget_chars:
            return truncated

        # Step 4: strip imported_papers metadata
        for paper in truncated.imported_papers:
            paper.metadata = {}
            paper.summary = (paper.summary or "")[:200]
        if _size(truncated) <= budget_chars:
            return truncated

        # Step 5: strip relevant_summaries to 3
        truncated.relevant_summaries = truncated.relevant_summaries[:3]
        if _size(truncated) <= budget_chars:
            return truncated

        # Step 6: cap session_history to 1 turn and strip remaining lists
        truncated.session_history = truncated.session_history[-1:]
        truncated.known_conclusions = truncated.known_conclusions[:3]
        truncated.open_questions = truncated.open_questions[:3]
        truncated.current_task_plan = truncated.current_task_plan[:2]
        truncated.imported_papers = truncated.imported_papers[:3]
        if _size(truncated) <= budget_chars:
            return truncated

        # Step 7: nuclear fallback — minimal context
        return ResearchContextSlice(
            research_topic=truncated.research_topic,
            research_goals=truncated.research_goals[:3],
            selected_papers=truncated.selected_papers[:8],
            metadata={"truncated": True, "fallback": True},
        )

    def _fallback_rule_decision(
        self,
        state: ResearchSupervisorState,
        *,
        planner_runs: int,
        replan_count: int,
        clarification_request: str | None,
        state_update: dict[str, Any],
    ) -> ResearchSupervisorDecision | None:
        if clarification_request:
            return self._fallback_clarify(
                state,
                thought="The manager needs the user to narrow the goal before continuing.",
                rationale="The current request still needs clarification, so it is safer to stop and ask for scope refinement.",
                planner_runs=planner_runs,
                replan_count=replan_count,
                clarification_request=clarification_request,
                state_update=state_update,
            )
        if state.workflow_constraint == "discovery_only":
            if state.last_action_name == "search_literature":
                if state.latest_result_status == "succeeded" and (state.has_report or state.paper_count > 0):
                    return self._fallback_finalize(
                        state,
                        stop_reason="Discovery-only workflow completed after literature search.",
                        thought="The constrained discovery pass already produced a usable search result.",
                        rationale="Search-only entry points should stop once the literature search result is available.",
                        planner_runs=planner_runs,
                        replan_count=replan_count,
                        clarification_request=None,
                        state_update=state_update,
                    )
                if state.latest_result_status == "failed":
                    return self._fallback_finalize(
                        state,
                        stop_reason="Discovery-only workflow stopped after literature search failed.",
                        thought="The constrained discovery pass cannot continue after the search step failed.",
                        rationale="Search-only entry points should stop instead of branching into unrelated actions after a failed discovery attempt.",
                        planner_runs=planner_runs,
                        replan_count=replan_count,
                        clarification_request=None,
                        state_update=state_update,
                    )
            return self._fallback_action_decision(
                action_name="search_literature",
                state=state,
                thought="This workflow is constrained to a single discovery pass.",
                rationale="Search-only entry points should execute literature search directly and stop after the result is produced.",
                phase="act",
                estimated_gain=0.96,
                estimated_cost=0.18,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if self._needs_goal_clarification(state):
            clarification = "当前研究目标还比较宽泛，请补充更具体的子方向、任务场景、方法类型或评价维度。"
            return self._fallback_clarify(
                state,
                thought="The research goal is too broad to launch a high-signal literature search.",
                rationale="A narrower topic will produce a stronger candidate set and reduce wasted search/import work.",
                planner_runs=planner_runs,
                replan_count=replan_count,
                clarification_request=clarification,
                state_update=state_update,
            )
        if state.mode == "document" and state.document_understood:
            return self._fallback_finalize(
                state,
                stop_reason="The uploaded document has already been parsed and indexed for later research use.",
                thought="The document-specific request is complete.",
                rationale="Document mode should stop after the uploaded file is turned into reusable evidence.",
                planner_runs=planner_runs,
                replan_count=replan_count,
                clarification_request=None,
                state_update=state_update,
            )
        if state.mode == "chart" and state.chart_understood:
            return self._fallback_finalize(
                state,
                stop_reason="The uploaded chart has already been understood and stored as structured evidence.",
                thought="The chart-specific request is complete.",
                rationale="Chart mode should stop after the visual evidence has been extracted.",
                planner_runs=planner_runs,
                replan_count=replan_count,
                clarification_request=None,
                state_update=state_update,
            )
        if self._should_understand_document(state):
            return self._fallback_action_decision(
                action_name="understand_document",
                state=state,
                thought="The uploaded document should be parsed before broader research actions.",
                rationale="Document understanding converts the raw file into reusable evidence for later QA and retrieval.",
                phase="act",
                estimated_gain=0.95,
                estimated_cost=0.35,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if self._should_understand_chart(state):
            return self._fallback_action_decision(
                action_name="supervisor_understand_chart",
                state=state,
                thought="The uploaded chart should be structured before broader research actions.",
                rationale="Chart understanding turns the visual artifact into evidence that later workers can cite.",
                phase="act",
                estimated_gain=0.88,
                estimated_cost=0.30,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if self._should_recommend_from_preferences(state):
            return self._fallback_action_decision(
                action_name="recommend_from_preferences",
                state=state,
                thought="The user wants broad worth-reading suggestions, so the manager should route to the long-term preference specialist.",
                rationale="Preference-based recommendation should use the dedicated agent that can read the user's persistent interest profile across sessions.",
                phase="reflect",
                estimated_gain=0.86,
                estimated_cost=0.18,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if state.preference_recommendation_requested and state.known_interest_count == 0:
            clarification = "我还没有积累足够的长期兴趣画像。你可以先连续问几个关心的论文主题，或直接告诉我想长期关注哪些方向。"
            return self._fallback_clarify(
                state,
                thought="The user asked for preference-based recommendations, but no stable long-term interests have been learned yet.",
                rationale="A short clarification is safer than pretending the system already knows the user's persistent interests.",
                planner_runs=planner_runs,
                replan_count=replan_count,
                clarification_request=clarification,
                state_update=state_update,
            )
        if not state.has_task:
            return self._fallback_action_decision(
                action_name="search_literature",
                state=state,
                thought="A discovery pass is needed to create the initial research workspace.",
                rationale="Without a task and candidate paper pool, the manager has no grounded context for later actions.",
                phase="plan",
                estimated_gain=1.0,
                estimated_cost=0.40,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if (
            state.last_action_name == "search_literature"
            and not state.paper_analysis_requested
            and "write_review" not in state.failed_actions
        ):
            return self._fallback_action_decision(
                action_name="write_review",
                state=state,
                thought="The manager should condense the fresh discovery results into a grounded review before the next step.",
                rationale="A short synthesis improves the workspace and matches the expected research handoff after discovery.",
                phase="reflect",
                estimated_gain=0.92,
                estimated_cost=0.36,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if self._should_compress_context(state) and (
            state.paper_analysis_requested
            or state.force_context_compression
        ):
            return self._fallback_action_decision(
                action_name="compress_context",
                state=state,
                thought="The workspace should be compressed before deeper synthesis tasks.",
                rationale="Selected-paper analysis quality improves when the manager hands workers a denser shared summary.",
                phase="plan",
                estimated_gain=0.54,
                estimated_cost=0.12,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if self._should_analyze(state):
            return self._fallback_action_decision(
                action_name="analyze_papers",
                state=state,
                thought="The workspace is ready for a focused selected-paper analysis.",
                rationale="There are enough papers in scope and the user explicitly asked for comparison, explanation, or recommendation-oriented analysis.",
                phase="reflect",
                estimated_gain=0.72,
                estimated_cost=0.24,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if self._should_import(state):
            return self._fallback_action_decision(
                action_name="import_papers",
                state=state,
                thought="The manager should ground the strongest papers into imported evidence.",
                rationale="Importing PDF-backed papers unlocks grounded QA and moves the workspace from discovery to evidence-backed research.",
                phase="act",
                estimated_gain=0.82,
                estimated_cost=0.45,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if self._should_answer(state):
            return self._fallback_action_decision(
                action_name="answer_question",
                state=state,
                thought="The workspace is ready to answer the user's current question.",
                rationale="Imported evidence or the active QA mode indicates that collection QA will provide the highest next value.",
                phase="reflect",
                estimated_gain=0.78,
                estimated_cost=0.28,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if self._should_compress_context(state):
            return self._fallback_action_decision(
                action_name="compress_context",
                state=state,
                thought="The workspace has grown enough that a compressed summary will help future reasoning.",
                rationale="Context compression is the safest remaining deterministic step before stopping.",
                phase="plan",
                estimated_gain=0.54,
                estimated_cost=0.12,
                planner_runs=planner_runs,
                replan_count=replan_count,
                state_update=state_update,
            )
        if state.failed_actions:
            return self._fallback_finalize(
                state,
                stop_reason=f"The manager stopped after {state.failed_actions[-1]} failed and preserved the partial workspace for recovery.",
                thought="The last worker action failed, so the safest next step is to stop and return the partial workspace.",
                rationale="A failed tool action should surface as a recovery stop instead of being silently ignored.",
                planner_runs=planner_runs,
                replan_count=replan_count,
                clarification_request=None,
                state_update=state_update,
            )
        return None

    def _fallback_action_decision(
        self,
        *,
        action_name: ResearchSupervisorActionName,
        state: ResearchSupervisorState,
        thought: str,
        rationale: str,
        phase: Literal["plan", "act", "reflect", "commit"],
        estimated_gain: float,
        estimated_cost: float,
        planner_runs: int,
        replan_count: int,
        state_update: dict[str, Any],
    ) -> ResearchSupervisorDecision:
        worker_agent = self._worker_for_action(action_name, None)
        plan_id = f"fallback_plan_{uuid4().hex[:12]}"
        payload = self._default_payload_for_action(action_name, state)
        payload = self._normalize_payload_paper_scope(action_name=action_name, payload=payload, state=state)
        payload = self._normalize_supervisor_route_payload(action_name=action_name, payload=payload, state=state)
        instruction = self._default_instruction_for_action(action_name, state, payload)
        active_message = AgentMessage(
            task_id=f"fallback_task_{uuid4().hex[:12]}",
            agent_from="ResearchSupervisorAgent",
            agent_to=worker_agent,
            task_type=self._task_type_for_action(action_name),
            instruction=instruction,
            payload=payload,
            context_slice={},
            priority="medium",
            expected_output_schema=self._expected_output_schema_for_action(action_name),
            metadata={
                "plan_id": plan_id,
                "decision_source": "fallback_rules",
                "manager_action": action_name,
            },
        )
        agent_messages = [*state_update["agent_messages"], active_message]
        return self._decision(
            action_name=action_name,
            thought=thought,
            rationale=rationale,
            phase=phase,
            estimated_gain=estimated_gain,
            estimated_cost=estimated_cost,
            action_input={**payload, "instruction": instruction},
            metadata={
                "decision_source": "fallback_rules",
                "worker_agent": worker_agent,
                "worker_task_type": self._task_type_for_action(action_name),
                "plan_id": plan_id,
                "active_message": active_message,
                "state_update": {
                    **state_update,
                    "pending_agent_messages": [active_message],
                    "agent_messages": agent_messages,
                    "planner_runs": planner_runs + 1,
                    "replan_count": replan_count,
                    "clarification_request": None,
                    "active_plan_id": plan_id,
                },
            },
        )

    def _fallback_finalize(
        self,
        state: ResearchSupervisorState,
        *,
        stop_reason: str,
        thought: str,
        rationale: str,
        planner_runs: int,
        replan_count: int,
        clarification_request: str | None,
        state_update: dict[str, Any],
    ) -> ResearchSupervisorDecision:
        failed_task_ids = self._dedupe_ids(list(state_update.get("failed_agent_task_ids", [])))
        replanned_failure_task_ids = self._dedupe_ids(list(state_update.get("replanned_failure_task_ids", [])))
        recovered_failure_ids = self._dedupe_ids([*replanned_failure_task_ids, *failed_task_ids])
        recovery_increment = 1 if failed_task_ids and failed_task_ids != replanned_failure_task_ids else 0
        return self._decision(
            action_name="finalize",
            thought=thought,
            rationale=rationale,
            phase="commit",
            estimated_gain=0.0,
            estimated_cost=0.0,
            stop_reason=stop_reason,
            metadata={
                "decision_source": "fallback_rules",
                "worker_agent": "ResearchSupervisorAgent",
                "state_update": {
                    **state_update,
                    "pending_agent_messages": [],
                    "planner_runs": planner_runs,
                    "replan_count": replan_count + recovery_increment,
                    "clarification_request": clarification_request,
                    "active_plan_id": None,
                    "replanned_failure_task_ids": recovered_failure_ids,
                },
            },
        )

    def _fallback_clarify(
        self,
        state: ResearchSupervisorState,
        *,
        thought: str,
        rationale: str,
        planner_runs: int,
        replan_count: int,
        clarification_request: str,
        state_update: dict[str, Any],
    ) -> ResearchSupervisorDecision:
        return self._decision(
            action_name="clarify_request",
            thought=thought,
            rationale=rationale,
            phase="commit",
            estimated_gain=0.2,
            estimated_cost=0.02,
            action_input={
                "clarification_question": clarification_request,
                "route_mode": state.route_mode,
            },
            stop_reason=clarification_request,
            metadata={
                "decision_source": "fallback_rules",
                "worker_agent": "ResearchSupervisorAgent",
                "clarification_request": clarification_request,
                "state_update": {
                    **state_update,
                    "pending_agent_messages": [],
                    "planner_runs": planner_runs,
                    "replan_count": replan_count,
                    "clarification_request": clarification_request,
                    "active_plan_id": None,
                },
            },
        )

    def _needs_goal_clarification(self, state: ResearchSupervisorState) -> bool:
        if state.has_task or state.mode not in {"auto", "research"}:
            return False
        normalized = re.sub(r"\s+", " ", state.goal.strip().lower())
        if not normalized:
            return True
        generic_goals = {
            "ai",
            "a.i.",
            "ml",
            "llm",
            "nlp",
            "cv",
            "人工智能",
            "机器学习",
            "深度学习",
            "大模型",
        }
        if normalized in generic_goals:
            return True
        compact = normalized.replace(" ", "")
        return len(compact) <= 3 and not any(marker in compact for marker in ("无人机", "路径规划", "agent", "review"))

    def _looks_like_follow_up_question(self, goal: str) -> bool:
        normalized = goal.strip().lower()
        if not normalized:
            return False
        if "?" in normalized or "？" in normalized:
            return True
        question_markers = (
            "哪些",
            "哪个",
            "如何",
            "为什么",
            "是否",
            "能否",
            "能不能",
            "请问",
            "what",
            "which",
            "how",
            "why",
            "whether",
        )
        return any(marker in normalized for marker in question_markers)

    def _decision(
        self,
        *,
        action_name: ResearchSupervisorActionName,
        thought: str,
        rationale: str,
        phase: str,
        estimated_gain: float,
        estimated_cost: float,
        action_input: dict[str, Any] | None = None,
        stop_reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ResearchSupervisorDecision:
        return ResearchSupervisorDecision(
            action_name=action_name,
            thought=thought,
            rationale=rationale,
            phase=phase,
            estimated_gain=estimated_gain,
            estimated_cost=estimated_cost,
            stop_reason=stop_reason,
            action_input=action_input or {},
            metadata=metadata or {},
        )

    def _stop_reason(self, state: ResearchSupervisorState) -> str:
        if state.mode == "qa" and state.answer_attempted:
            return "The requested collection QA step has already been executed in this loop."
        if state.import_attempted and state.importable_paper_count == 0:
            return "No additional importable open-access paper remained after the import step."
        if state.open_todo_count > 0 and state.evidence_gap_count > 0:
            return "The workspace now has explicit evidence-gap TODOs, so the manager should hand control back to the user."
        return "No tool action offers enough additional value compared with the current research workspace state."

    def _dispatch_message(self, message: AgentMessage, state: ResearchSupervisorState) -> dict[str, Any]:
        if message.task_type == "search_and_review":
            return {
                "action_name": "search_literature",
                "worker_agent": "LiteratureScoutAgent",
                "thought": "Manager 先让 LiteratureScoutAgent 完成检索、过滤与研究空间初始化，再把结果交给后续 worker。",
                "rationale": "Discovery should be owned by a dedicated scout worker so the manager can coordinate workers directly without an extra sub-manager hop.",
                "phase": "plan",
                "estimated_gain": 1.0,
                "estimated_cost": 0.4,
                "action_input": {"goal": state.goal, "mode": state.mode},
            }
        if message.task_type == "search_literature":
            return {
                "action_name": "search_literature",
                "worker_agent": "LiteratureScoutAgent",
                "thought": "Manager 先分派检索与筛选任务给 LiteratureScoutAgent，再让下游写作 worker 处理综述成稿。",
                "rationale": "Discovery belongs in the scout worker because it fans out into search, ranking, and task materialization.",
                "phase": "plan",
                "estimated_gain": 1.0,
                "estimated_cost": 0.42,
                "action_input": {"goal": state.goal, "mode": state.mode},
            }
        if message.task_type == "write_review":
            return {
                "action_name": "write_review",
                "worker_agent": "ResearchWriterAgent",
                "thought": "Manager 把综述撰写交给 ResearchWriterAgent，由它负责成稿和内部质检。",
                "rationale": "Report drafting should be owned by the writing worker directly instead of being wrapped behind a writing sub-manager.",
                "phase": "reflect",
                "estimated_gain": 0.92,
                "estimated_cost": 0.36,
                "action_input": {"goal": state.goal, "mode": state.mode},
            }
        if message.task_type == "import_papers":
            return {
                "action_name": "import_papers",
                "worker_agent": "ResearchKnowledgeAgent",
                "thought": "Manager 把导入与证据落地交给 ResearchKnowledgeAgent，由它完成候选解析和 grounded ingestion。",
                "rationale": "Evidence grounding belongs in the knowledge worker because it couples candidate selection with downstream QA scope.",
                "phase": "act",
                "estimated_gain": 0.82,
                "estimated_cost": 0.45,
                "action_input": {
                    "selected_paper_count": state.selected_paper_count,
                    "import_top_k": state.import_top_k,
                    "auto_import": state.auto_import,
                    "importable_paper_count": state.importable_paper_count,
                },
        }
        if message.task_type == "answer_question":
            return {
                "action_name": "answer_question",
                "worker_agent": "ResearchQAAgent",
                "thought": "Manager 将问答交给 ResearchQAAgent，让它按 Supervisor 选择的 QA route 完成 grounded QA。",
                "rationale": "Task-level QA should be owned by the QA specialist while RAG and retrieval remain tool-layer capabilities.",
                "phase": "reflect",
                "estimated_gain": 0.78,
                "estimated_cost": 0.28,
                "action_input": {"goal": state.goal, "top_level_mode": state.mode},
            }
        if message.task_type == "recommend_from_preferences":
            return {
                "action_name": "recommend_from_preferences",
                "worker_agent": "PreferenceMemoryAgent",
                "thought": "Manager 把跨会话兴趣推荐交给 PreferenceMemoryAgent，由它基于长期记忆挑选最近值得看的论文。",
                "rationale": "Long-term preference modeling and personalized recommendation should live in a dedicated specialist instead of being hidden inside discovery or paper-analysis workers.",
                "phase": "reflect",
                "estimated_gain": 0.84,
                "estimated_cost": 0.18,
                "action_input": {"goal": state.goal, "top_level_mode": state.mode},
            }
        if message.task_type in {"compare_papers", "recommend_papers", "analyze_papers"}:
            return {
                "action_name": "analyze_papers",
                "worker_agent": "PaperAnalysisAgent",
                "thought": "Manager 把基于已选论文的分析交给 PaperAnalysisAgent，由统一的论文分析能力完成比较、讲解或推荐。",
                "rationale": "Selected-paper analysis should flow through one analysis capability so comparison, recommendation, and explanation share the same evidence and reasoning surface.",
                "phase": "reflect",
                "estimated_gain": 0.72,
                "estimated_cost": 0.24,
                "action_input": {"goal": state.goal, "top_level_mode": state.mode},
            }
        if message.task_type == "compress_context":
            return {
                "action_name": "compress_context",
                "worker_agent": "ResearchKnowledgeAgent",
                "thought": "Manager 先让 ResearchKnowledgeAgent 压缩当前研究上下文，减少后续 QA、对比和推荐的上下文噪声。",
                "rationale": "Context compression is a knowledge-side concern because it depends on paper cards, session history, and current workspace focus.",
                "phase": "plan",
                "estimated_gain": 0.54,
                "estimated_cost": 0.12,
                "action_input": {"goal": state.goal, "top_level_mode": state.mode},
            }
        if message.task_type == "understand_document":
            return {
                "action_name": "understand_document",
                "worker_agent": "ResearchDocumentAgent",
                "thought": "Planner 把文档理解排在最前面，先解析用户提供的文档证据。",
                "rationale": "Document understanding turns a raw file into indexed evidence that the research loop can reuse.",
                "phase": "act",
                "estimated_gain": 0.95,
                "estimated_cost": 0.35,
                "action_input": {"goal": state.goal, "mode": state.mode},
            }
        if message.task_type == "supervisor_understand_chart":
            return {
                "action_name": "supervisor_understand_chart",
                "worker_agent": "ChartAnalysisAgent",
                "thought": "Planner 识别到图表证据输入，先完成图表结构化理解以服务后续研究分析。",
                "rationale": "Chart understanding belongs inside the research assistant as a visual evidence worker.",
                "phase": "act",
                "estimated_gain": 0.88,
                "estimated_cost": 0.3,
                "action_input": {"goal": state.goal, "mode": state.mode},
            }
        raise ValueError(f"unsupported planner task_type: {message.task_type}")

    def _extend_messages(
        self,
        existing: list[AgentMessage],
        new_messages: list[AgentMessage],
    ) -> list[AgentMessage]:
        by_id = {message.task_id: message for message in existing}
        for message in new_messages:
            by_id[message.task_id] = message
        return list(by_id.values())

    def _select_ready_message(
        self,
        pending: list[AgentMessage],
        *,
        completed_task_ids: list[str],
    ) -> AgentMessage | None:
        completed = set(completed_task_ids)
        for message in pending:
            if set(message.dependencies).issubset(completed):
                return message
        return None

    def _latest_unreplanned_failure(
        self,
        results: list[AgentResultMessage],
        *,
        replanned_failure_task_ids: list[str],
    ) -> AgentResultMessage | None:
        replanned = set(replanned_failure_task_ids)
        for result in reversed(results):
            evaluation_failed = result.evaluation is not None and not result.evaluation.passed
            if (result.status == "failed" or evaluation_failed) and result.task_id not in replanned:
                return result
        return None

    def _dedupe_ids(self, values: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            deduped.append(value)
            seen.add(value)
        return deduped

    def _evaluate_results(
        self,
        *,
        results: list[AgentResultMessage],
        agent_messages: list[AgentMessage],
    ) -> list[AgentResultMessage]:
        message_by_task_id = {message.task_id: message for message in agent_messages}
        evaluated: list[AgentResultMessage] = []
        for result in results:
            if result.evaluation is not None:
                evaluated.append(result)
                continue
            source_message = message_by_task_id.get(result.task_id)
            evaluation = self._evaluate_result(
                result=result,
                task_instruction=source_message.instruction if source_message is not None else result.instruction,
                expected_schema=(
                    source_message.expected_output_schema
                    if source_message is not None
                    else result.expected_output_schema
                ),
            )
            evaluated.append(result.model_copy(update={"evaluation": evaluation}))
        return evaluated

    def _evaluate_result(
        self,
        *,
        result: AgentResultMessage,
        task_instruction: str,
        expected_schema: dict[str, Any],
    ) -> TaskEvaluation:
        return self.evaluation_skill.evaluate_result(
            task_type=result.task_type,
            result_status=result.status,
            payload={
                **result.payload,
                **({"reason": result.metadata.get("reason")} if result.metadata.get("reason") else {}),
            },
            task_instruction=task_instruction,
            expected_schema=expected_schema,
        )
