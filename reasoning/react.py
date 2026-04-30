"""Backward-compatibility shim — canonical code lives in agents.research_qa_agent."""

from __future__ import annotations

from agents.research_qa_agent import (  # noqa: F401
    ReActDecision,
    ReActFinalDraft,
    ReActStep,
    ResearchQAAgent,
    ResearchQAAgentError,
)

# Legacy aliases so existing callers keep working
ReActReasoningAgent = ResearchQAAgent
ReActReasoningAgentError = ResearchQAAgentError
