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
    "ResearchAgentContextBuilder",
    "ResearchAgentResultAggregator",
    "ResearchCapabilityRegistry",
    "ResearchExternalToolGateway",
    "ResearchFunctionService",
    "ResearchKnowledgeAccess",
    "ResearchMemoryGateway",
    "ResearchReportService",
    "ResearchSkillResolver",
    "ResearchSkillSelection",
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
    "PaperImportService": "tools.research.paper_import",
    "PaperSearchService": "tools.research.paper_search",
    "PaperSelectorService": "tools.research.paper_selector",
    "ResearchExecutionContext": "domain.schemas.research_context",
    "ResearchContextManager": "memory.research_context_manager",
    "ResearchAgentContextBuilder": "runtime.research.context_builder",
    "ResearchAgentResultAggregator": "runtime.research.result_aggregator",
    "ResearchCapabilityRegistry": "tools.research.capability_registry",
    "ResearchExternalToolGateway": "tools.research.external_tool_gateway",
    "ResearchFunctionService": "tools.research.research_functions",
    "ResearchKnowledgeAccess": "tools.research.knowledge_access",
    "ResearchMemoryGateway": "memory.research_memory_gateway",
    "ResearchReportService": "adapters.storage.research_report_service",
    "ResearchSkillResolver": "tools.research.skill_resolver",
    "ResearchSkillSelection": "tools.research.skill_resolver",
    "ResearchSupervisorGraphRuntime": "runtime.research.supervisor_graph_runtime",
    "UnifiedAgentRegistry": "runtime.research.unified_runtime",
    "UnifiedRuntimeContext": "runtime.research.unified_runtime",
    "build_phase1_unified_agent_registry": "runtime.research.unified_runtime",
    "build_phase1_unified_blueprint": "runtime.research.unified_runtime",
    "serialize_unified_agent_messages": "runtime.research.unified_runtime",
    "serialize_unified_agent_results": "runtime.research.unified_runtime",
    "serialize_unified_agent_registry": "runtime.research.unified_runtime",
    "serialize_unified_delegation_plan": "runtime.research.unified_runtime",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
