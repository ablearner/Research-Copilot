from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AnswerTools",
    "AnswerToolsError",
    "ChartTools",
    "ChartToolsError",
    "DocumentTools",
    "DocumentToolsError",
    "GraphExtractionTools",
    "GraphExtractionToolsError",
    "PageSummary",
    "PageSummaryInput",
    "RetrievalAgentResult",
    "RetrievalTools",
    "RetrievalToolsError",
]

_EXPORT_MAP = {
    "AnswerTools": ("tools.answer_toolkit", "AnswerTools"),
    "AnswerToolsError": ("tools.answer_toolkit", "AnswerToolsError"),
    "ChartTools": ("tools.chart_toolkit", "ChartTools"),
    "ChartToolsError": ("tools.chart_toolkit", "ChartToolsError"),
    "DocumentTools": ("tools.document_toolkit", "DocumentTools"),
    "DocumentToolsError": ("tools.document_toolkit", "DocumentToolsError"),
    "GraphExtractionTools": ("tools.graph_extraction_toolkit", "GraphExtractionTools"),
    "GraphExtractionToolsError": ("tools.graph_extraction_toolkit", "GraphExtractionToolsError"),
    "PageSummary": ("tools.document_toolkit", "PageSummary"),
    "PageSummaryInput": ("tools.graph_extraction_toolkit", "PageSummaryInput"),
    "RetrievalAgentResult": ("tools.retrieval_toolkit", "RetrievalAgentResult"),
    "RetrievalTools": ("tools.retrieval_toolkit", "RetrievalTools"),
    "RetrievalToolsError": ("tools.retrieval_toolkit", "RetrievalToolsError"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

