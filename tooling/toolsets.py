"""Toolset definitions for different Kepler platforms.

Groups tools by capability and platform so that different frontends
(CLI, API, MCP) can select the appropriate tool subset.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

TOOLSETS: dict[str, dict[str, Any]] = {
    "research-core": {
        "description": "Core research tools",
        "tools": ["hybrid_retrieve", "query_graph_summary", "answer_with_evidence"],
    },
    "document": {
        "description": "Document processing tools",
        "tools": ["parse_document", "understand_chart"],
    },
    "academic-search": {
        "description": "Academic search APIs",
        "tools": ["arxiv_search", "semantic_scholar_search", "openalex_search", "ieee_search"],
    },
    "memory": {
        "description": "Memory management tools",
        "tools": ["update_user_profile", "record_conclusion"],
    },
    "kepler-cli": {
        "description": "Full CLI toolset",
        "tools": [],
        "includes": ["research-core", "document", "academic-search", "memory"],
    },
    "kepler-api": {
        "description": "API server toolset",
        "tools": [],
        "includes": ["research-core", "document", "academic-search", "memory"],
    },
    "kepler-mcp": {
        "description": "MCP server toolset (read-only)",
        "tools": [],
        "includes": ["research-core", "academic-search"],
    },
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
