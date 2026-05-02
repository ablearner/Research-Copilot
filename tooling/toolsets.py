"""Toolset definitions for different Kepler platforms.

Groups tools by capability and platform so that different frontends
(CLI, API, MCP) can select the appropriate tool subset.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TOOLSETS: dict[str, dict[str, Any]] = {
    "rag": {
        "description": "Core RAG pipeline tools",
        "tools": [
            "parse_document", "index_document", "understand_chart",
            "hybrid_retrieve", "query_graph_summary", "answer_with_evidence",
            "ask_document", "ask_fused", "graph_backfill_document",
        ],
    },
    "research_function": {
        "description": "Research high-level functions",
        "tools": [
            "search_papers", "extract_paper_structure", "compare_papers",
            "generate_review", "ask_paper", "recommend_papers",
            "update_research_context", "decompose_task",
            "evaluate_result", "execute_research_plan",
        ],
    },
    "supervisor_action": {
        "description": "Supervisor decision actions",
        "tools": [
            "search_literature", "write_review", "import_papers",
            "sync_to_zotero", "answer_question", "general_answer",
            "recommend_from_preferences", "analyze_papers",
            "compress_context", "understand_document",
            "supervisor_understand_chart", "analyze_paper_figures",
        ],
    },
    "runtime_ops": {
        "description": "Runtime operation tools",
        "tools": [
            "academic_search", "local_file", "code_execution",
            "web_search", "notification", "library_sync",
            "search_or_import_paper",
        ],
    },
    "academic-search": {
        "description": "Academic search APIs",
        "tools": ["arxiv_search", "semantic_scholar_search", "openalex_search", "ieee_search"],
    },
    "memory": {
        "description": "Memory management tools",
        "tools": ["update_user_profile", "record_conclusion"],
    },
    "research-capability": {
        "description": "Research capability tools (ranking, writing, analysis, intent)",
        "tools": [
            "paper_ranking", "paper_reading", "paper_analysis",
            "paper_curation", "paper_chart_analysis",
            "survey_writing", "review_writing", "writing_polish",
            "query_planning", "user_intent", "research_evaluation",
            "visual_anchor", "qa_routing", "code_linking",
        ],
    },
    "research-core": {
        "description": "Core research tools (alias for rag subset)",
        "tools": ["hybrid_retrieve", "query_graph_summary", "answer_with_evidence"],
    },
    "document": {
        "description": "Document processing tools",
        "tools": ["parse_document", "understand_chart"],
    },
    "kepler-cli": {
        "description": "Full CLI toolset",
        "tools": [],
        "includes": ["rag", "research_function", "runtime_ops", "academic-search", "memory"],
    },
    "kepler-api": {
        "description": "API server toolset",
        "tools": [],
        "includes": ["rag", "research_function", "runtime_ops", "academic-search", "memory"],
    },
    "kepler-mcp": {
        "description": "MCP server toolset (read-only)",
        "tools": [],
        "includes": ["research-core", "academic-search"],
    },
}

AGENT_TOOLSET_MAP: dict[str, list[str]] = {
    "ResearchSupervisorAgent": ["supervisor_action", "research-capability"],
    "ResearchQAAgent": ["rag", "research-capability"],
    "ResearchDocumentAgent": ["rag"],
    "ChartAnalysisAgent": ["rag", "research-capability"],
    "LiteratureScoutAgent": ["research_function", "runtime_ops", "research-capability"],
    "ResearchKnowledgeAgent": ["research_function", "runtime_ops"],
    "ResearchWriterAgent": ["research_function", "research-capability"],
    "PaperAnalysisAgent": ["research_function", "research-capability"],
    "GeneralAnswerAgent": ["rag", "research_function"],
    "PreferenceMemoryAgent": ["research_function"],
}


def resolve_toolset(name: str, *, _visited: set[str] | None = None) -> list[str]:
    """Recursively resolve a toolset name to a flat list of tool names."""
    if _visited is None:
        _visited = set()
    if name in _visited:
        return []
    _visited.add(name)

    toolset = TOOLSETS.get(name)
    if toolset is None:
        logger.warning("Unknown toolset: %s", name)
        return []

    tools: list[str] = list(toolset.get("tools", []))
    for included in toolset.get("includes", []):
        tools.extend(resolve_toolset(included, _visited=_visited))

    return list(dict.fromkeys(tools))


def resolve_agent_toolsets(agent_name: str) -> list[str]:
    """Return the list of toolset names an agent is allowed to use."""
    return list(AGENT_TOOLSET_MAP.get(agent_name, []))
