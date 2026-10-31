"""Prepare isolated specialist data with explicit tool and service capabilities."""

from copy import deepcopy
from dataclasses import fields, replace
from typing import Any

from agents.research_supervisor_agent import ResearchSupervisorDecision
from runtime.research.agent_protocol.base import ResearchAgentToolContext
from runtime.research.unified_action_adapters import resolve_active_message
from runtime.research.worker_task_input import build_worker_task_input
from runtime.research.worker_ports import build_worker_knowledge_port, build_worker_research_port


# Exclude live resources from data copying; rebuild restricted ports below.
_LIVE_RESOURCE_FIELDS = frozenset({
    "research_service", "graph_runtime", "unified_agent_registry", "knowledge_access",
    "progress_callback",
})


def prepare_worker_invocation(
    *,
    context: ResearchAgentToolContext,
    decision: ResearchSupervisorDecision,
    agent_results: list[Any],
) -> tuple[ResearchAgentToolContext, ResearchSupervisorDecision]:
    """Snapshot task data; the runtime alone applies returned state deltas.

    This isolates input mutation, not external effects or concurrent commits.
    The supervisor graph intentionally continues to execute workers serially.
    """
    worker = replace(context, **deepcopy({
        item.name: getattr(context, item.name)
        for item in fields(context)
        if item.name not in _LIVE_RESOURCE_FIELDS and item.name != "result_artifacts"
    }), result_artifacts={})
    worker.research_service = build_worker_research_port(context.research_service)
    worker.graph_runtime = build_worker_knowledge_port(context.graph_runtime)
    worker.knowledge_access = worker.graph_runtime.knowledge_access
    worker.unified_agent_registry = None
    # Supervisor graph updates have already been applied before dispatch.
    # Workers need the task, not copies of pending tasks and the complete plan.
    task_decision = replace(
        decision,
        action_input=deepcopy(decision.action_input),
        metadata=deepcopy({key: value for key, value in decision.metadata.items() if key != "state_update"}),
    )
    worker.worker_input, worker.worker_artifacts = build_worker_task_input(
        context=context, decision=task_decision, agent_results=agent_results,
    )
    message = resolve_active_message(task_decision)
    # A task without a Skill must not inherit the previous invocation's body.
    metadata = message.metadata if message is not None else task_decision.metadata
    worker.supervisor_instruction = worker.worker_input.instruction or None
    worker.skill_context = str(metadata.get("skill_context") or "").strip() or None
    if worker.skill_selection is not None:
        worker.skill_selection.active_skill_names = list(metadata.get("active_skill_names") or [])
        worker.skill_selection.skill_context = worker.skill_context or ""
        if not worker.skill_selection.active_skill_names:
            worker.skill_selection.required_tools = []
            worker.skill_selection.missing_tools = {}
    if message is not None:
        task_decision.metadata["active_message"] = message.model_copy(
            update={"context_slice": worker.worker_input.context_slice.model_copy(deep=True)},
            deep=True,
        )
    return worker, task_decision
