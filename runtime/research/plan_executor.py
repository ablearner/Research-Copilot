"""Pure plan validation and state transitions; dispatch belongs to the runtime."""
from __future__ import annotations

from uuid import uuid4

from domain.schemas.agent_message import AgentResultMessage
from domain.schemas.research_plan import PlanStep, ResearchPlan

_ACCEPTED = {"done", "skipped"}


def _validate_steps(
    steps: list[PlanStep], *, allowed_actions: set[str],
    workflow_constraint: str | None, completed_ids: set[int] | None = None,
) -> list[PlanStep]:
    completed_ids = completed_ids or set()
    ids = [step.step_id for step in steps]
    if len(ids) != len(set(ids)) or completed_ids.intersection(ids):
        raise ValueError("Plan step IDs must be unique and cannot replace completed steps")
    known_ids = set(ids) | completed_ids
    for step in steps:
        if step.action not in allowed_actions or step.action in {
            "finalize", "clarify", "clarify_request",
        }:
            raise ValueError(f"Unsupported plan action: {step.action}")
        if workflow_constraint == "discovery_only" and step.action not in {"search_literature", "write_review"}:
            raise ValueError("discovery_only plans may only search_literature and write_review")
        if step.step_id in step.depends_on or not set(step.depends_on) <= known_ids:
            raise ValueError(f"Invalid dependencies for step {step.step_id}")
    resolved = set(completed_ids)
    remaining = list(steps)
    while remaining:
        eligible = [step for step in remaining if set(step.depends_on) <= resolved]
        if not eligible:
            raise ValueError("Plan dependencies contain a cycle")
        resolved.update(step.step_id for step in eligible)
        remaining = [step for step in remaining if step.step_id not in resolved]
    copied = [step.model_copy(deep=True) for step in steps]
    for step in copied:
        step.status = "pending"
        step.task_id = None
        step.attempts = 0
    return copied


def create_plan(
    steps: list[PlanStep], *, allowed_actions: set[str],
    workflow_constraint: str | None = None, plan_id: str | None = None,
) -> ResearchPlan:
    if not steps:
        raise ValueError("A plan must contain at least one step")
    validated = _validate_steps(
        steps, allowed_actions=allowed_actions, workflow_constraint=workflow_constraint,
    )
    return ResearchPlan(plan_id=plan_id or str(uuid4()), steps=validated)


def ready_step(plan: ResearchPlan) -> PlanStep | None:
    if plan.status != "active" or any(step.status == "running" for step in plan.steps):
        return None
    completed = {step.step_id for step in plan.steps if step.status in _ACCEPTED}
    return next((step for step in plan.steps
                 if step.status == "pending" and set(step.depends_on) <= completed), None)


def start_step(plan: ResearchPlan, step_id: int, task_id: str) -> ResearchPlan:
    ready = ready_step(plan)
    if ready is None or ready.step_id != step_id or not task_id.strip():
        raise ValueError("Only the next ready step can be started with a nonempty task ID")
    if any(
        step.status in _ACCEPTED and step.action == ready.action and step.params == ready.params
        for step in plan.steps
    ):
        raise ValueError(f"Step {step_id} repeats a completed action with identical params")
    used_task_ids = {step.task_id for step in plan.steps if step.task_id}
    for event in plan.history:
        if event.get("event") == "step_result":
            used_task_ids.add(event["result"]["task_id"])
        elif event.get("event") == "plan_replaced":
            used_task_ids.update(step.get("task_id") for step in event["steps"])
    if task_id in used_task_ids:
        raise ValueError("Each dispatch requires a fresh task ID")
    updated = plan.model_copy(deep=True)
    step = next(step for step in updated.steps if step.step_id == step_id)
    step.status = "running"
    step.task_id = task_id
    step.attempts += 1
    return updated


def advance_plan(plan: ResearchPlan, results: list[AgentResultMessage]) -> ResearchPlan:
    updated = plan.model_copy(deep=True)
    if updated.status != "active":
        return updated
    for step in updated.steps:
        if step.status != "running":
            continue
        result = next((result for result in results
                       if result.task_id == step.task_id
                       and result.status in {"succeeded", "failed", "skipped"}), None)
        if result is None:
            return updated
        passed = result.evaluation is None or result.evaluation.passed
        accepted_skip = (result.status == "skipped" and step.allow_skip
                         and result.evaluation is not None and result.evaluation.passed)
        updated.history.append({
            "event": "step_result", "revision": updated.revision,
            "step_id": step.step_id, "result": result.model_dump(mode="json"),
        })
        if (result.status == "succeeded" and passed) or accepted_skip:
            step.status = "skipped" if accepted_skip else "done"
        else:
            step.status = "failed"
            updated.status = "needs_replan"
            issues = "; ".join(result.evaluation.issues) if result.evaluation else ""
            updated.reason = f"Step {step.step_id} ({step.action}) was not accepted: {result.status}"
            if result.evaluation is not None and not result.evaluation.passed:
                updated.reason += "; evaluation failed"
            if issues:
                updated.reason += f"; {issues}"
            return updated
    if all(step.status in _ACCEPTED for step in updated.steps):
        updated.status = "completed"
        updated.reason = ""
    elif not any(step.status == "running" for step in updated.steps) and ready_step(updated) is None:
        updated.status = "blocked"
        updated.reason = "No pending step has satisfied dependencies"
    return updated


def replace_remaining_plan(
    plan: ResearchPlan, steps: list[PlanStep], *, allowed_actions: set[str],
    workflow_constraint: str | None = None,
) -> ResearchPlan:
    completed = [step for step in plan.steps if step.status in _ACCEPTED]
    if not steps and not completed:
        raise ValueError("A replacement plan must contain at least one step")
    validated = _validate_steps(
        steps, allowed_actions=allowed_actions, workflow_constraint=workflow_constraint,
        completed_ids={step.step_id for step in completed},
    )
    for step in validated:
        if any(step.action == done.action and step.params == done.params for done in completed):
            raise ValueError(f"Step {step.step_id} repeats a completed action with identical params")
    updated = plan.model_copy(deep=True)
    updated.history.append({
        "event": "plan_replaced", "revision": plan.revision, "reason": plan.reason,
        "steps": [step.model_dump(mode="json") for step in plan.steps
                  if step.status not in _ACCEPTED],
    })
    updated.steps = [step for step in updated.steps if step.status in _ACCEPTED] + validated
    updated.revision += 1
    updated.status = "active" if validated else "completed"
    updated.reason = ""
    return updated
