"""Explicit worker capabilities; dispatch and application commits stay in Runtime.

These are logical application boundaries, not a Python security sandbox. Bound
callbacks retain their implementation resources without exposing their owners as
part of the worker API.
"""
from dataclasses import dataclass
from typing import Any

from tools.research.knowledge_access import ResearchKnowledgeAccess


_KNOWLEDGE_TOOLS = frozenset({
    "hybrid_retrieve", "query_graph_summary", "answer_with_evidence",
    "parse_document", "graph_backfill_document", "index_document",
    "ask_document", "ask_fused", "understand_chart",
})


@dataclass(frozen=True, slots=True)
class WorkerToolRegistry:
    get_tool: Any


@dataclass(frozen=True, slots=True)
class WorkerToolExecutor:
    execute_tool_call: Any


@dataclass(frozen=True, slots=True)
class WorkerZoteroPort:
    sync_paper_to_zotero: Any


@dataclass(frozen=True, slots=True)
class WorkerReportPort:
    load_papers: Any
    load_report: Any


@dataclass(frozen=True, slots=True)
class WorkerImportPort:
    prepare_paper_import: Any


@dataclass(frozen=True, slots=True)
class WorkerQAPort:
    execute_qa: Any


@dataclass(frozen=True, slots=True)
class WorkerChartPort:
    llm_adapter: Any


@dataclass(slots=True)
class WorkerKnowledgePort:
    retrieval_tools: Any
    answer_tools: Any
    document_tools: Any
    chart_tools: Any
    session_memory: Any
    research_function_service: WorkerZoteroPort | None
    tool_registry: WorkerToolRegistry | None
    tool_executor: WorkerToolExecutor | None
    query_graph_summary: Any
    handle_parse_document: Any
    handle_graph_backfill_document: Any
    handle_index_document: Any
    handle_ask_document: Any
    handle_ask_fused: Any
    handle_understand_chart: Any
    knowledge_access: Any = None


@dataclass(frozen=True, slots=True)
class WorkerResearchPort:
    create_task: Any
    build_execution_context: Any
    build_context_slices: Any
    append_runtime_event: Any
    get_task: Any
    list_paper_figures: Any
    analyze_paper_figure: Any
    _load_cached_figure_payload: Any
    report_service: WorkerReportPort
    paper_service: WorkerImportPort
    qa_service: WorkerQAPort
    paper_curation_tool: Any
    research_context_manager: Any
    paper_reading_tool: Any
    llm_adapter: Any
    visual_intent_router: Any


def build_worker_knowledge_port(runtime: Any) -> WorkerKnowledgePort:
    registry = getattr(runtime, "tool_registry", None)
    executor = getattr(runtime, "tool_executor", None)

    def get_tool(name, *, include_disabled=False):
        if name not in _KNOWLEDGE_TOOLS:
            return None
        return registry.get_tool(name, include_disabled=include_disabled)

    async def execute_tool_call(name, payload):
        if name not in _KNOWLEDGE_TOOLS:
            raise ValueError(f"Tool is not available to research workers: {name}")
        return await executor.execute_tool_call(name, payload)

    functions = getattr(runtime, "research_function_service", None)
    charts = getattr(runtime, "chart_tools", None)
    port = WorkerKnowledgePort(
        retrieval_tools=getattr(runtime, "retrieval_tools", None),
        answer_tools=getattr(runtime, "answer_tools", None),
        document_tools=getattr(runtime, "document_tools", None),
        chart_tools=WorkerChartPort(getattr(charts, "llm_adapter", None)) if charts is not None else None,
        session_memory=getattr(runtime, "session_memory", None),
        research_function_service=WorkerZoteroPort(functions.sync_paper_to_zotero)
        if callable(getattr(functions, "sync_paper_to_zotero", None)) else None,
        tool_registry=WorkerToolRegistry(get_tool) if registry is not None else None,
        tool_executor=WorkerToolExecutor(execute_tool_call) if executor is not None else None,
        query_graph_summary=getattr(runtime, "query_graph_summary", None),
        handle_parse_document=getattr(runtime, "handle_parse_document", None),
        handle_graph_backfill_document=getattr(runtime, "handle_graph_backfill_document", None),
        handle_index_document=getattr(runtime, "handle_index_document", None),
        handle_ask_document=getattr(runtime, "handle_ask_document", None),
        handle_ask_fused=getattr(runtime, "handle_ask_fused", None),
        handle_understand_chart=getattr(runtime, "handle_understand_chart", None),
    )
    port.knowledge_access = ResearchKnowledgeAccess.from_runtime(port)
    return port


def build_worker_research_port(service: Any) -> WorkerResearchPort:
    reports = getattr(service, "report_service", None)
    papers = getattr(service, "paper_service", None)
    qa = getattr(service, "qa_service", None)

    async def create_task(request, **kwargs):
        if request.run_immediately:
            raise ValueError("Workers may create task records but cannot dispatch a research run")
        return await service.create_task(request, **kwargs)

    async def execute_qa(task_id, request, *, graph_runtime, persist=False, executor=None):
        if persist or executor is None:
            raise ValueError("Worker QA requires its current executor and deferred persistence")
        return await qa.execute_qa(
            task_id, request, graph_runtime=graph_runtime, persist=False, executor=executor,
        )

    return WorkerResearchPort(
        create_task=create_task,
        build_execution_context=getattr(service, "build_execution_context", None),
        build_context_slices=getattr(service, "build_context_slices", None),
        append_runtime_event=getattr(service, "append_runtime_event", None),
        get_task=getattr(service, "get_task", None),
        list_paper_figures=getattr(service, "list_paper_figures", None),
        analyze_paper_figure=getattr(service, "analyze_paper_figure", None),
        _load_cached_figure_payload=getattr(service, "_load_cached_figure_payload", None),
        report_service=WorkerReportPort(getattr(reports, "load_papers", None), getattr(reports, "load_report", None)),
        paper_service=WorkerImportPort(getattr(papers, "prepare_paper_import", None)),
        qa_service=WorkerQAPort(execute_qa),
        paper_curation_tool=getattr(service, "paper_curation_tool", None),
        research_context_manager=getattr(service, "research_context_manager", None),
        paper_reading_tool=getattr(service, "paper_reading_tool", None),
        llm_adapter=getattr(service, "llm_adapter", None),
        visual_intent_router=getattr(service, "visual_intent_router", None),
    )
