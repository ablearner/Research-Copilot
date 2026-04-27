from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "LiteratureResearchService",
    "PaperImportService",
    "PaperSearchService",
    "PaperSelectorService",
    "ResearchExecutionContext",
    "ResearchContextManager",
    "ResearchFunctionService",
    "ResearchReportService",
    "ResearchSupervisorGraphRuntime",
    "UnifiedAgentRegistry",
    "UnifiedRuntimeContext",
    "build_phase1_unified_agent_registry",
    "build_phase1_unified_blueprint",
    "serialize_unified_agent_messages",
    "serialize_unified_agent_results",
    "serialize_unified_agent_registry",
    "serialize_unified_delegation_plan",
]

_EXPORTS = {
    "LiteratureResearchService": "services.research.literature_research_service",
    "PaperImportService": "services.research.paper_import_service",
    "PaperSearchService": "services.research.paper_search_service",
    "PaperSelectorService": "services.research.paper_selector_service",
    "ResearchExecutionContext": "services.research.research_context",
    "ResearchContextManager": "services.research.research_context_manager",
    "ResearchFunctionService": "services.research.research_function_service",
    "ResearchReportService": "services.research.research_report_service",
    "ResearchSupervisorGraphRuntime": "services.research.research_supervisor_graph_runtime",
    "UnifiedAgentRegistry": "services.research.unified_runtime",
    "UnifiedRuntimeContext": "services.research.unified_runtime",
    "build_phase1_unified_agent_registry": "services.research.unified_runtime",
    "build_phase1_unified_blueprint": "services.research.unified_runtime",
    "serialize_unified_agent_messages": "services.research.unified_runtime",
    "serialize_unified_agent_results": "services.research.unified_runtime",
    "serialize_unified_agent_registry": "services.research.unified_runtime",
    "serialize_unified_delegation_plan": "services.research.unified_runtime",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
