"""Reasoning strategy layer.

This package holds reusable reasoning strategies and style normalization used by:

- answer synthesis (`CoTReasoningAgent`)
- query decomposition (`PlanAndSolveReasoningAgent`)
- optional tool-using answer flow (`ReActReasoningAgent`)

Unlike `planners/`, this is not a thin routing helper layer. These modules
contain real LLM-facing reasoning behaviors that are still actively used.
"""

from reasoning.cot import CoTReasoningAgent
from reasoning.plan_and_solve import PlanAndSolveReasoningAgent
from reasoning.react import ReActReasoningAgent, ReActReasoningAgentError
from reasoning.strategies import (
    AnswerSynthesisStrategy,
    QueryPlanningStrategy,
    ReasoningStrategySet,
    ToolReasoningStrategy,
)
from reasoning.style import normalize_reasoning_style

__all__ = [
    "CoTReasoningAgent",
    "PlanAndSolveReasoningAgent",
    "ReActReasoningAgent",
    "ReActReasoningAgentError",
    "AnswerSynthesisStrategy",
    "QueryPlanningStrategy",
    "ReasoningStrategySet",
    "ToolReasoningStrategy",
    "normalize_reasoning_style",
]
