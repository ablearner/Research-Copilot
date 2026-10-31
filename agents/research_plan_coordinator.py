"""Adapt pure plan transitions to existing supervisor dispatch and evaluation."""
from __future__ import annotations

from typing import TYPE_CHECKING, get_args
from uuid import uuid4

from domain.schemas.research_plan import ResearchPlan
from runtime.research.plan_executor import (
    advance_plan, create_plan, ready_step, replace_remaining_plan, start_step,
)

if TYPE_CHECKING:
    from agents.research_supervisor_agent import ResearchSupervisorAgent


class ResearchPlanCoordinator:
    def __init__(self, supervisor: ResearchSupervisorAgent, *, max_replans: int = 2):
        if max_replans < 0:
            raise ValueError("max_plan_replans must be nonnegative")
        self.supervisor = supervisor
        self.max_replans = max_replans

    def stop(self, *, state, reason, all_messages, results, planner_runs, replan_count, context_slice=None):
        if state.execution_plan is not None and state.execution_plan.status != "completed":
            state.execution_plan = state.execution_plan.model_copy(update={"status": "blocked", "reason": reason}, deep=True)
        return self.supervisor._guardrail_finalize(
            state, all_messages=all_messages, results=results, planner_runs=planner_runs,
            replan_count=replan_count, reason=reason,
        )

    def advance(self, *, state, results, **kwargs):
        if state.execution_plan is None:
            return None
        state.execution_plan = advance_plan(state.execution_plan, results)
        plan = state.execution_plan
        if plan.status == "completed":
            return self.stop(state=state, results=results, reason="All plan steps passed acceptance checks.", **kwargs)
        if plan.status == "failed":
            return self.stop(state=state, results=results, reason=plan.reason, **kwargs)
        if plan.status in {"blocked", "needs_replan"}:
            if plan.replan_attempts >= self.max_replans:
                return self.stop(state=state, results=results,
                                 reason=f"Plan replan limit ({self.max_replans}) reached: {plan.reason}", **kwargs)
            return None
        if ready_step(plan) is None:
            return self.stop(state=state, results=results, reason="Waiting for the running plan task result.", **kwargs)
        return self.dispatch(state=state, results=results, **kwargs)

    def accept(self, *, llm_output, state, **kwargs):
        from agents.research_supervisor_agent import ResearchSupervisorActionName

        try:
            if not llm_output.plan:
                return self.stop(state=state, reason=llm_output.stop_reason or "No executable replacement plan was supplied.", **kwargs)
            allowed_actions = set(get_args(ResearchSupervisorActionName))
            threshold_keys = {
                "min_paper_count", "min_report_words", "min_imported_or_skipped",
                "min_answer_chars", "min_evidence_count", "min_synced_count", "min_summary_count",
            }
            boolean_keys = {"require_citations", "require_key_sections", "require_no_failures"}
            for step in llm_output.plan:
                if step.action not in allowed_actions:
                    raise ValueError(f"Step {step.step_id}: unsupported action {step.action}")
                paper_ids = step.params.get("paper_ids")
                if paper_ids is not None and (not isinstance(paper_ids, list) or any(not isinstance(item, str) or not item.strip() for item in paper_ids)):
                    raise ValueError(f"Step {step.step_id}: paper_ids must be a list of nonempty strings")
                route = step.params.get("qa_route")
                if route is not None and (not isinstance(route, str) or route not in {"collection_qa", "document_drilldown", "chart_drilldown"}):
                    raise ValueError(f"Step {step.step_id}: invalid qa_route")
                for key in ("top_k", "import_top_k", "days_back"):
                    value = step.params.get(key)
                    if value is not None and (type(value) is not int or value < (0 if key == "import_top_k" else 1)):
                        raise ValueError(f"Step {step.step_id}: invalid {key}")
                descriptive_schema = self.supervisor._expected_output_schema_for_action(step.action)
                for key, value in step.expected_output_schema.items():
                    if key not in threshold_keys | boolean_keys | {"required_fields", "required_paper_ids"} | descriptive_schema.keys():
                        raise ValueError(f"Step {step.step_id}: unsupported acceptance key {key}")
                    if key in threshold_keys and (type(value) is not int or value < 0):
                        raise ValueError(f"Step {step.step_id}: {key} must be a nonnegative integer")
                    if key in {"required_fields", "required_paper_ids"} and (not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value)):
                        raise ValueError(f"Step {step.step_id}: {key} must be a list of nonempty strings")
                    if key in boolean_keys and type(value) is not bool:
                        raise ValueError(f"Step {step.step_id}: {key} must be boolean")
                    if key in descriptive_schema and value != descriptive_schema[key]:
                        raise ValueError(f"Step {step.step_id}: descriptive field {key} must retain its declared type")
            options = dict(allowed_actions=set(get_args(ResearchSupervisorActionName)), workflow_constraint=state.workflow_constraint)
            if state.execution_plan is None:
                state.execution_plan = create_plan(llm_output.plan, **options)
            else:
                state.execution_plan = replace_remaining_plan(state.execution_plan, llm_output.plan, **options)
        except ValueError as exc:
            if state.execution_plan is None:
                state.execution_plan = ResearchPlan(plan_id=f"invalid_{uuid4().hex[:12]}", steps=[], status="failed")
            return self.stop(state=state, reason=f"Invalid execution plan: {exc}", **kwargs)
        return self.dispatch(state=state, **kwargs)

    def dispatch(self, *, state, **kwargs):
        from agents.research_supervisor_agent import ResearchSupervisorLLMDecision

        plan = state.execution_plan
        step = ready_step(plan)
        if step is None:
            return self.stop(state=state, reason="No executable plan step remains.", **kwargs)
        blocked = self.supervisor._skill_policy_block_reason(step.action, state)
        if blocked:
            return self.stop(state=state, reason=blocked, **kwargs)
        # Reuse the existing action boundary for both the first and later steps.
        decision = self.supervisor._decision_from_llm_output(
            llm_output=ResearchSupervisorLLMDecision(
                action_name=step.action, instruction=step.instruction, payload=step.params,
                expected_output_schema=self._acceptance_schema(step),
                thought=f"Execute plan step {step.step_id}: {step.action}",
                rationale=step.instruction or f"Execute plan step {step.step_id}",
            ), state=state, **kwargs,
        )
        message = decision.metadata["active_message"]
        if message.payload.get("unresolved_paper_refs") or message.payload.get("dropped_paper_refs"):
            return self.stop(state=state, reason=f"Unresolved paper scope in plan step {step.step_id}; no worker dispatched.", **kwargs)
        paper_ids = message.payload.get("paper_ids") or []
        if paper_ids and step.action in {"analyze_papers", "answer_question", "import_papers", "sync_to_zotero"}:
            message.expected_output_schema["required_paper_ids"] = list(dict.fromkeys([
                *message.expected_output_schema.get("required_paper_ids", []), *paper_ids,
            ]))
        if step.action == "import_papers" and paper_ids:
            message.expected_output_schema["min_imported_or_skipped"] = max(
                len(set(paper_ids)), message.expected_output_schema["min_imported_or_skipped"],
            )
        if step.action == "sync_to_zotero" and paper_ids:
            message.expected_output_schema["min_synced_count"] = max(
                len(set(paper_ids)), message.expected_output_schema["min_synced_count"],
            )
        message.metadata.update(plan_id=plan.plan_id, plan_step_id=step.step_id, plan_revision=plan.revision,
                                decision_source="plan_executor")
        message.depends_on = [prior.task_id for prior in plan.steps if prior.step_id in step.depends_on and prior.task_id]
        decision.metadata.update(plan_id=plan.plan_id, decision_source="plan_executor", plan_step_id=step.step_id)
        try:
            state.execution_plan = start_step(plan, step.step_id, message.task_id)
        except ValueError as exc:
            return self.stop(state=state, reason=f"Plan dispatch rejected: {exc}", **kwargs)
        return decision

    def _acceptance_schema(self, step):
        schema = dict(self.supervisor._expected_output_schema_for_action(step.action))
        criteria = {
            "search_literature": {"min_paper_count": 1},
            "import_papers": {"min_imported_or_skipped": 1},
            "sync_to_zotero": {
                "min_synced_count": 1, "require_no_failures": True,
                "required_fields": ["synced_count", "failed_count"],
            },
            "write_review": {"required_fields": ["report_id"], "min_report_words": 1},
            "answer_question": {"min_evidence_count": 1},
            "analyze_papers": {"min_paper_count": 1, "required_fields": ["analyzed_paper_ids", "analysis_focus"]},
            "analyze_paper_figures": {"required_fields": ["answer"]},
            "general_answer": {"required_fields": ["answer"]},
            "compress_context": {"min_summary_count": 1},
            "recommend_from_preferences": {"required_fields": ["recommendations"]},
        }
        schema.update(criteria.get(step.action, {}))
        # A proposal can strengthen, but cannot erase, the action's minimum requirements.
        for key, value in step.expected_output_schema.items():
            if key == "required_fields":
                schema[key] = list(dict.fromkeys([*schema.get(key, []), *value]))
            elif key.startswith("min_") and key in schema:
                schema[key] = max(schema[key], value)
            elif key == "require_no_failures":
                schema[key] = schema.get(key, False) or value
            else:
                schema[key] = value
        return schema
