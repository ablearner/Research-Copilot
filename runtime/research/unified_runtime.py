from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from domain.schemas.unified_runtime import (
    UnifiedAgentDescriptor,
    UnifiedAgentResult,
    UnifiedAgentTask,
    UnifiedExecutionMode,
    UnifiedRuntimeBlueprint,
    UnifiedCapabilityBinding,
)


ACTION_AGENT_CATALOG: dict[str, str] = {
    "search_literature": "LiteratureScoutAgent",
    "write_review": "ResearchWriterAgent",
    "import_papers": "ResearchKnowledgeAgent",
    "sync_to_zotero": "ResearchKnowledgeAgent",
    "answer_question": "ResearchAnalystAgent",
    "general_answer": "ResearchAnalystAgent",
    "recommend_from_preferences": "LiteratureScoutAgent",
    "analyze_papers": "ResearchAnalystAgent",
    "compress_context": "ResearchKnowledgeAgent",
    "understand_document": "ResearchKnowledgeAgent",
    "supervisor_understand_chart": "ResearchAnalystAgent",
    "analyze_paper_figures": "ResearchAnalystAgent",
}


def _actions_for_worker(name: str) -> list[str]:
    return [action for action, owner in ACTION_AGENT_CATALOG.items() if owner == name]


class UnifiedAgentExecutor(Protocol):
    descriptor: UnifiedAgentDescriptor

    async def execute_action(self, context: Any, decision: Any, *, action_name: str) -> Any:
        ...


class UnifiedAgentRegistry:
    def __init__(self) -> None:
        self._executors: dict[str, UnifiedAgentExecutor] = {}

    def register(self, executor: UnifiedAgentExecutor, replace: bool = False) -> UnifiedAgentExecutor:
        name = executor.descriptor.name
        if name in self._executors and not replace:
            raise RuntimeError(f"Unified agent already registered: {name}")
        self._executors[name] = executor
        return executor

    def get(self, name: str) -> UnifiedAgentExecutor | None:
        return self._executors.get(name)

    def list(self) -> list[UnifiedAgentExecutor]:
        return list(self._executors.values())

    def resolve_for_task(self, task_type: str) -> list[UnifiedAgentExecutor]:
        return [
            executor
            for executor in self._executors.values()
            if task_type in executor.descriptor.supported_task_types
        ]

    def resolve_for_action(self, action_name: str) -> UnifiedAgentExecutor | None:
        agent_name = ACTION_AGENT_CATALOG.get(action_name)
        return self.get(agent_name) if agent_name else None


@dataclass(slots=True)
class SpecialistAgentExecutor:
    descriptor: UnifiedAgentDescriptor
    agent_delegate: Any

    async def execute_action(self, context: Any, decision: Any, *, action_name: str) -> Any:
        if action_name not in self.descriptor.supported_task_types:
            raise ValueError(f"{self.descriptor.name} cannot execute {action_name}")
        kwargs = {"task_type": action_name} if len(self.descriptor.supported_task_types) > 1 else {}
        return await self.agent_delegate.run_action(context, decision, **kwargs)


def build_phase1_unified_blueprint(
    *,
    graph_runtime: Any,
    research_service: Any,
) -> UnifiedRuntimeBlueprint:
    del research_service
    tool_names = _tool_names_from_runtime(graph_runtime)
    capability_profile_names = _capability_profile_names_from_runtime(graph_runtime)
    agent_descriptors = [
        _build_descriptor(
            name="ResearchSupervisorAgent",
            description="Supervisor that plans, routes, and evaluates research tasks.",
            kind="orchestrator",
            execution_mode="hybrid",
            supported_task_types=["plan", "replan", "finalize", "delegate"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_supervisor", capability_profile_names),
                service_names=["ResearchEvaluationTool"],
                notes=[
                    "Plans and routes tasks; evaluates specialist results.",
                ],
            ),
            preferred_tool_names=[
                "decompose_task",
                "evaluate_result",
                "update_research_context",
                "recommend_papers",
                "search_or_import_paper",
                "parse_document",
                "index_document",
            ],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=[
                "Internal task and result state uses AgentMessage and AgentResultMessage.",
            ],
        ),
        _build_descriptor(
            name="LiteratureScoutAgent",
            description="Literature discovery and personalized recommendations from research preferences.",
            kind="specialist",
            execution_mode="hybrid",
            supported_task_types=_actions_for_worker("LiteratureScoutAgent"),
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                service_names=["PaperSearchService", "ResearchMemoryGateway"],
            ),
            preferred_tool_names=["search_papers", "academic_search", "search_or_import_paper"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Owns query planning, search, curation and preference recommendations."],
        ),
        _build_descriptor(
            name="ResearchKnowledgeAgent",
            description="Knowledge ingestion, document parsing/indexing, library sync and context compression.",
            kind="specialist",
            execution_mode="tool_native",
            supported_task_types=_actions_for_worker("ResearchKnowledgeAgent"),
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
            ),
            preferred_tool_names=["search_or_import_paper", "parse_document", "index_document",
                                  "hybrid_retrieve", "query_graph_summary"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Document processing is an internal workflow, not an independent worker."],
        ),
        _build_descriptor(
            name="ResearchAnalystAgent",
            description="Grounded QA, paper comparison, chart/figure understanding and general answers.",
            kind="specialist",
            execution_mode="hybrid",
            supported_task_types=_actions_for_worker("ResearchAnalystAgent"),
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                service_names=["PaperAnalysisTool", "PaperReadingTool"],
            ),
            preferred_tool_names=["collection_qa", "ask_document", "ask_fused", "hybrid_retrieve",
                                  "query_graph_summary", "answer_with_evidence", "understand_chart",
                                  "extract_paper_structure", "compare_papers"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Supervisor selects the task and QA route; internal workflows do not dispatch peers."],
        ),
        _build_descriptor(
            name="ResearchWriterAgent",
            description="Review/report synthesis, quality checks and TODO generation.",
            kind="specialist",
            execution_mode="service_native",
            supported_task_types=_actions_for_worker("ResearchWriterAgent"),
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                service_names=["ReviewWritingTool", "WritingPolishTool"],
            ),
            preferred_tool_names=["generate_review"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Owns synthesis over the evidence supplied by Supervisor."],
        ),
    ]
    return UnifiedRuntimeBlueprint(
        name="research-copilot-unified-runtime",
        agent_descriptors=agent_descriptors,
        tool_names=tool_names,
        capability_profile_names=capability_profile_names,
        unresolved_boundaries=[],
        migration_stages=[
            "Current: Supervisor plans and routes; specialists execute capability-backed tasks; runtime owns lifecycle and events.",
        ],
    )


def build_phase1_unified_agent_registry(
    *,
    graph_runtime: Any,
    research_service: Any,
    agent_delegates: dict[str, Any] | None = None,
    blueprint: UnifiedRuntimeBlueprint | None = None,
) -> UnifiedAgentRegistry:
    blueprint = blueprint or build_phase1_unified_blueprint(
        graph_runtime=graph_runtime,
        research_service=research_service,
    )
    registry = UnifiedAgentRegistry()
    delegates = agent_delegates or {}
    for descriptor in blueprint.agent_descriptors:
        delegate = delegates.get(descriptor.name)
        if descriptor.kind == "specialist" and delegate is not None and hasattr(delegate, "run_action"):
            registry.register(SpecialistAgentExecutor(descriptor=descriptor, agent_delegate=delegate))
    return registry


def serialize_unified_agent_registry(registry: UnifiedAgentRegistry | None) -> list[dict[str, Any]]:
    if registry is None:
        return []
    return [executor.descriptor.model_dump(mode="json") for executor in registry.list()]


def serialize_agent_exchange(
    agent_messages: list[Any],
    agent_results: list[Any],
    *,
    registry: UnifiedAgentRegistry | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Project internal task messages to the public legacy and unified fields."""
    results_by_id = {result.task_id: result for result in agent_results}
    legacy_plan = []
    for message in agent_messages:
        result = results_by_id.get(message.task_id)
        legacy_plan.append({
            "task_id": message.task_id,
            "assigned_to": message.agent_to,
            "task_type": message.task_type,
            "instruction": message.instruction,
            "depends_on": list(message.depends_on),
            "priority": message.priority,
            "retry_count": message.retry_count,
            "status": _projected_status(result),
            "evaluation": result.evaluation.model_dump(mode="json") if result and result.evaluation else None,
        })
    return {
        "agent_messages": [message.model_dump(mode="json") for message in agent_messages],
        "agent_results": [result.model_dump(mode="json") for result in agent_results],
        "delegation_plan": legacy_plan,
        "unified_agent_messages": serialize_unified_agent_messages(agent_messages, registry=registry),
        "unified_agent_results": serialize_unified_agent_results(agent_results, registry=registry),
        "unified_delegation_plan": serialize_unified_delegation_plan(agent_messages, agent_results, registry=registry),
    }


def _projected_status(result: Any | None) -> str:
    if result is None:
        return "planned"
    return "failed" if result.evaluation is not None and not result.evaluation.passed else result.status


def serialize_unified_agent_messages(
    agent_messages: list[Any],
    *,
    registry: UnifiedAgentRegistry | None = None,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in agent_messages:
        descriptor = _descriptor_for_agent(registry, message.agent_to)
        task = UnifiedAgentTask.from_agent_message(
            message,
            preferred_skill_name=_preferred_capability_name(
                metadata=getattr(message, "metadata", {}),
                descriptor=descriptor,
            ),
            available_tool_names=_available_tool_names(descriptor),
        )
        payload = task.model_dump(mode="json")
        if descriptor is not None:
            payload["agent_descriptor"] = descriptor.model_dump(mode="json")
        serialized.append(payload)
    return serialized


def serialize_unified_agent_results(
    agent_results: list[Any],
    *,
    registry: UnifiedAgentRegistry | None = None,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for result in agent_results:
        if isinstance(result, UnifiedAgentResult):
            unified_result = result
        else:
            unified_result = UnifiedAgentResult.from_agent_result_message(result)
        payload = unified_result.model_dump(mode="json")
        descriptor = _descriptor_for_agent(registry, unified_result.agent_name)
        preferred_skill_name = _preferred_capability_name(
            metadata=unified_result.metadata,
            descriptor=descriptor,
        )
        if preferred_skill_name:
            payload["preferred_skill_name"] = preferred_skill_name
        if descriptor is not None:
            payload["available_tool_names"] = _available_tool_names(descriptor)
            payload["execution_mode"] = descriptor.execution_mode
            payload["agent_descriptor"] = descriptor.model_dump(mode="json")
        serialized.append(payload)
    return serialized


def serialize_unified_delegation_plan(
    agent_messages: list[Any],
    agent_results: list[Any],
    *,
    registry: UnifiedAgentRegistry | None = None,
) -> list[dict[str, Any]]:
    result_by_task_id: dict[str, UnifiedAgentResult] = {}
    for result in agent_results:
        unified_result = result if isinstance(result, UnifiedAgentResult) else UnifiedAgentResult.from_agent_result_message(result)
        result_by_task_id[unified_result.task_id] = unified_result
    serialized: list[dict[str, Any]] = []
    for message in agent_messages:
        descriptor = _descriptor_for_agent(registry, message.agent_to)
        task = UnifiedAgentTask.from_agent_message(
            message,
            preferred_skill_name=_preferred_capability_name(
                metadata=getattr(message, "metadata", {}),
                descriptor=descriptor,
            ),
            available_tool_names=_available_tool_names(descriptor),
        )
        payload = task.model_dump(mode="json")
        result = result_by_task_id.get(message.task_id)
        payload["status"] = _projected_status(result)
        payload["evaluation"] = (
            result.evaluation.model_dump(mode="json")
            if result is not None and result.evaluation is not None
            else None
        )
        payload["action_output"] = (
            dict(result.action_output)
            if result is not None and result.action_output is not None
            else None
        )
        if descriptor is not None:
            payload["agent_descriptor"] = descriptor.model_dump(mode="json")
        serialized.append(payload)
    return serialized


def _build_descriptor(
    *,
    name: str,
    description: str,
    kind: str,
    execution_mode: UnifiedExecutionMode,
    supported_task_types: list[str],
    capability_binding: UnifiedCapabilityBinding,
    preferred_tool_names: list[str],
    available_tool_names: list[str],
    legacy_boundaries: list[str],
    notes: list[str],
) -> UnifiedAgentDescriptor:
    available = [tool_name for tool_name in preferred_tool_names if tool_name in set(available_tool_names)]
    if kind == "specialist":
        supported_task_types = [action for action, agent in ACTION_AGENT_CATALOG.items() if agent == name]
    return UnifiedAgentDescriptor(
        name=name,
        description=description,
        kind=kind,
        execution_mode=execution_mode,
        supported_task_types=supported_task_types,
        capability_binding=capability_binding,
        preferred_tool_names=preferred_tool_names,
        available_tool_names=available,
        legacy_boundaries=legacy_boundaries,
        notes=notes,
    )


def _tool_names_from_runtime(graph_runtime: Any) -> list[str]:
    registry = getattr(graph_runtime, "tool_registry", None)
    if registry is None or not hasattr(registry, "list_tools"):
        return []
    try:
        return sorted(
            tool.name
            for tool in registry.list_tools(include_disabled=False)
            if getattr(tool, "name", None)
        )
    except Exception:
        return []


def _capability_profile_names_from_runtime(graph_runtime: Any) -> list[str]:
    return []


def _preferred_capability(name: str, available_skill_names: list[str]) -> str | None:
    if not name:
        return None
    if not available_skill_names:
        return name
    if name in set(available_skill_names):
        return name
    return None


def _descriptor_for_agent(
    registry: UnifiedAgentRegistry | None,
    agent_name: str,
) -> UnifiedAgentDescriptor | None:
    if registry is None:
        return None
    executor = registry.get(agent_name)
    if executor is None:
        return None
    return executor.descriptor


def _available_tool_names(descriptor: UnifiedAgentDescriptor | None) -> list[str]:
    if descriptor is None:
        return []
    return list(descriptor.available_tool_names)


def _preferred_capability_name(
    *,
    metadata: dict[str, Any] | None,
    descriptor: UnifiedAgentDescriptor | None,
) -> str | None:
    values = metadata or {}
    for key in ("preferred_skill_name", "skill_name"):
        raw_value = values.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()
    if descriptor is not None and descriptor.capability_binding.profile_name:
        return descriptor.capability_binding.profile_name
    return None
