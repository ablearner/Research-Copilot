# ruff: noqa: F401
"""Shared protocols, context types and utility mixins for the research agent graph runtime."""

from runtime.research.agent_protocol.base import (
    MemoryOp,
    ResearchAgentGraphState,
    ResearchAgentTool,
    ResearchAgentToolContext,
    ResearchStateDelta,
    ResearchToolResult,
    STATE_DELTA_METADATA_KEY,
    SpecialistAgent,
    _llm_stage_timeout_seconds,
    _message,
    _now_iso,
    _observation_envelope,
    _should_fallback_llm_stage,
    _update_runtime_progress,
)
from runtime.research.agent_protocol.mixins import (
    comparison_scope_papers,
    persist_workspace_results,
)
