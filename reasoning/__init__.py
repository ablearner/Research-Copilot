"""Backward-compatibility shim for reasoning workers.

Canonical code:
- ReAct reasoning → agents.research_qa_agent.RagReActQAWorker
- Plan-and-Execute → internal to LiteratureScoutAgent / ResearchKnowledgeAgent
- CoT → removed (AnswerChain used directly)
"""

from agents.research_qa_agent import (  # noqa: F401
    RagReActQAWorker,
    ResearchQAAgent,
    ResearchQAAgentError,
    normalize_reasoning_style,
)
from reasoning.cot import CoTReasoningAgent  # noqa: F401
from reasoning.plan_and_solve import PlanAndSolveReasoningAgent  # noqa: F401
from reasoning.strategies import ReasoningStrategySet  # noqa: F401

# Legacy aliases
ReActReasoningAgent = RagReActQAWorker
ReActReasoningAgentError = ResearchQAAgentError

__all__ = [
    "CoTReasoningAgent",
    "PlanAndSolveReasoningAgent",
    "ReActReasoningAgent",
    "ReActReasoningAgentError",
    "ReasoningStrategySet",
    "RagReActQAWorker",
    "ResearchQAAgent",
    "ResearchQAAgentError",
    "normalize_reasoning_style",
]
