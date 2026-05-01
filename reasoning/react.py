"""Backward-compatibility shim for the RAG-layer ReAct QA worker."""

from __future__ import annotations

from agents.research_qa_agent import (  # noqa: F401
    RagReActQAWorker,
    ReActDecision,
    ReActFinalDraft,
    ReActStep,
    ResearchQAAgent,
    ResearchQAAgentError,
)

# Legacy aliases so existing callers keep working
ReActReasoningAgent = RagReActQAWorker
ReActReasoningAgentError = ResearchQAAgentError
