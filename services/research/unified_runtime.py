from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from domain.schemas.unified_runtime import (
    UnifiedAgentDescriptor,
    UnifiedAgentResult,
    UnifiedAgentTask,
    UnifiedExecutionMode,
    UnifiedRuntimeBlueprint,
    UnifiedCapabilityBinding,
)


@dataclass(slots=True)
class UnifiedRuntimeContext:
    graph_runtime: Any
    research_service: Any
    tool_registry: Any | None = None
    tool_executor: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class UnifiedAgentExecutor(Protocol):
    descriptor: UnifiedAgentDescriptor

    async def execute(self, task: UnifiedAgentTask, context: UnifiedRuntimeContext) -> UnifiedAgentResult:
        ...


UnifiedExecutionHandler = Callable[
    [UnifiedAgentTask, UnifiedRuntimeContext, Any | None],
    Awaitable[UnifiedAgentResult],
]


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


@dataclass(slots=True)
class PhaseOneUnifiedAgentAdapter:
    descriptor: UnifiedAgentDescriptor
    agent_delegate: Any | None = None
    execution_handler: UnifiedExecutionHandler | None = None

    async def execute(self, task: UnifiedAgentTask, context: UnifiedRuntimeContext) -> UnifiedAgentResult:
        if self.execution_handler is not None:
            return await self.execution_handler(task, context, self.agent_delegate)
        del context
        return UnifiedAgentResult(
            task_id=task.task_id,
            agent_name=self.descriptor.name,
            task_type=task.task_type,
            status="skipped",
            instruction=task.instruction,
            payload={},
            context_slice=task.context_slice,
            priority=task.priority,
            expected_output_schema=task.expected_output_schema,
            depends_on=task.depends_on,
            retry_count=task.retry_count,
            metadata={
                "migration_phase": "phase1_skeleton",
                "reason": "legacy runtime still owns execution; this adapter only standardizes composition metadata",
                "delegate_type": (
                    self.agent_delegate.__class__.__name__ if self.agent_delegate is not None else None
                ),
            },
        )


def build_phase1_unified_runtime_context(
    *,
    graph_runtime: Any,
    research_service: Any,
) -> UnifiedRuntimeContext:
    return UnifiedRuntimeContext(
        graph_runtime=graph_runtime,
        research_service=research_service,
        tool_registry=getattr(graph_runtime, "tool_registry", None),
        tool_executor=getattr(graph_runtime, "tool_executor", None),
    )


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
            description="High-level orchestrator that should become a pure scheduler over a unified agent envelope.",
            kind="orchestrator",
            execution_mode="hybrid",
            supported_task_types=["plan", "replan", "finalize", "delegate"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_supervisor", capability_profile_names),
                service_names=["ResearchEvaluator"],
                notes=[
                    "Currently owns planning plus evaluation responsibilities.",
                    "Target state is scheduler-only with evaluation moved to a normal tool or reviewer agent.",
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
            legacy_boundaries=[
                "research_supervisor_graph_runtime_core.ResearchAgentTool",
                "manager-specific handoff logic",
            ],
            notes=[
                "This is the main boundary to collapse into the shared agent protocol.",
            ],
        ),
        _build_descriptor(
            name="LiteratureScoutAgent",
            description="Discovery specialist for topic planning, provider search, and import candidate scouting.",
            kind="specialist",
            execution_mode="hybrid",
            supported_task_types=["search_literature"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                service_names=["PaperSearchService"],
                notes=[
                    "Topic planning is currently embedded in the search service and reasoning agent.",
                ],
            ),
            preferred_tool_names=["search_papers", "academic_search", "search_or_import_paper"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Executes literature discovery through LiteratureDiscoveryCapability under the specialist boundary."],
        ),
        _build_descriptor(
            name="ResearchKnowledgeAgent",
            description="Knowledge ingestion and retrieval specialist over imported papers, graph summaries, and collection context.",
            kind="specialist",
            execution_mode="tool_native",
            supported_task_types=["import_papers", "sync_to_zotero", "compress_context"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                notes=["Owns import, library sync, retrieval-support, and context compression capabilities."],
            ),
            preferred_tool_names=[
                "search_or_import_paper",
                "hybrid_retrieve",
                "query_graph_summary",
                "answer_with_evidence",
                "ask_paper",
                "parse_document",
                "index_document",
            ],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Use this agent for knowledge operations and retrieval-support steps, not task-level QA routing."],
        ),
        _build_descriptor(
            name="ResearchDocumentAgent",
            description="Document understanding specialist for parsing uploaded files and indexing grounded evidence.",
            kind="specialist",
            execution_mode="tool_native",
            supported_task_types=["understand_document"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                notes=["Owns parse/index capability behind a single specialist boundary."],
            ),
            preferred_tool_names=["parse_document", "index_document"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Supervisor should route uploaded document grounding here, not to raw document tools."],
        ),
        _build_descriptor(
            name="ResearchQAAgent",
            description="Task-level QA specialist for supervisor-selected collection, document, and chart question answering routes.",
            kind="specialist",
            execution_mode="tool_native",
            supported_task_types=["answer_question"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                notes=["Supervisor selects qa_route; this agent executes the chosen QA tool path."],
            ),
            preferred_tool_names=[
                "collection_qa",
                "ask_document",
                "ask_fused",
                "hybrid_retrieve",
                "query_graph_summary",
                "answer_with_evidence",
            ],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Owns QA execution under Supervisor; RAG remains a tool layer behind ResearchKnowledgeAccess."],
        ),
        _build_descriptor(
            name="ResearchWriterAgent",
            description="Synthesis specialist for review drafting, answer polishing, and TODO generation.",
            kind="specialist",
            execution_mode="service_native",
            supported_task_types=["write_review"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                service_names=["ReviewWriter", "WritingPolisher", "PaperAnalyzer"],
                notes=["Capability classes live in services.research.capabilities."],
            ),
            preferred_tool_names=["generate_review", "answer_with_evidence", "compare_papers", "recommend_papers"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Executes review drafting through ReviewWritingCapability."],
        ),
        _build_descriptor(
            name="PaperAnalysisAgent",
            description="Single-paper drilldown specialist for structure extraction and reading cards.",
            kind="specialist",
            execution_mode="service_native",
            supported_task_types=["analyze_papers"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_report", capability_profile_names),
                service_names=["PaperAnalyzer", "PaperReader"],
            ),
            preferred_tool_names=["extract_paper_structure", "ask_paper", "parse_document", "index_document"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Executes selected-paper analysis through PaperAnalysisCapability."],
        ),
        _build_descriptor(
            name="ChartAnalysisAgent",
            description="Visual evidence specialist for chart understanding and chart-grounded answers.",
            kind="specialist",
            execution_mode="tool_native",
            supported_task_types=["understand_chart", "analyze_paper_figures"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("financial_report", capability_profile_names)
                or _preferred_capability("research_report", capability_profile_names),
            ),
            preferred_tool_names=["understand_chart"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Executes chart and paper-figure actions through ChartAnalysisCapability."],
        ),
        _build_descriptor(
            name="GeneralAnswerAgent",
            description="General-answer specialist for questions that do not require research workspace evidence.",
            kind="specialist",
            execution_mode="service_native",
            supported_task_types=["general_answer"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("general_answer", capability_profile_names),
                notes=["Executes direct answers through GeneralAnswerCapability."],
            ),
            preferred_tool_names=[],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Can request Supervisor reroute when the question needs research evidence."],
        ),
        _build_descriptor(
            name="PreferenceMemoryAgent",
            description="Preference-memory specialist for personalized paper recommendations.",
            kind="specialist",
            execution_mode="service_native",
            supported_task_types=["recommend_from_preferences"],
            capability_binding=UnifiedCapabilityBinding(
                profile_name=_preferred_capability("research_preferences", capability_profile_names),
                service_names=["ResearchMemoryGateway"],
                notes=["Long-term memory access is mediated by ResearchMemoryGateway."],
            ),
            preferred_tool_names=["search_papers", "academic_search"],
            available_tool_names=tool_names,
            legacy_boundaries=[],
            notes=["Executes preference recommendations through PreferenceRecommendationCapability."],
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
    execution_handlers: dict[str, UnifiedExecutionHandler] | None = None,
) -> UnifiedAgentRegistry:
    blueprint = build_phase1_unified_blueprint(
        graph_runtime=graph_runtime,
        research_service=research_service,
    )
    registry = UnifiedAgentRegistry()
    delegates = agent_delegates or {}
    handlers = execution_handlers or {}
    for descriptor in blueprint.agent_descriptors:
        registry.register(
            PhaseOneUnifiedAgentAdapter(
                descriptor=descriptor,
                agent_delegate=delegates.get(descriptor.name),
                execution_handler=handlers.get(descriptor.name),
            )
        )
    return registry


def serialize_unified_agent_registry(registry: UnifiedAgentRegistry | None) -> list[dict[str, Any]]:
    if registry is None:
        return []
    return [executor.descriptor.model_dump(mode="json") for executor in registry.list()]


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
        payload["status"] = (
            "planned"
            if result is None
            else (
                "failed"
                if result.evaluation is not None and not result.evaluation.passed
                else result.status
            )
        )
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
