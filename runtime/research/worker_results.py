"""Keep substantive worker outputs outside automatic model context."""

from copy import deepcopy
from typing import Any

from runtime.research.agent_protocol.base import ResearchToolResult


def worker_result_artifact(result: ResearchToolResult) -> tuple[dict[str, Any], str]:
    """Extract the actual output, plus a short preview for dependent tasks.

    Several action metadata contracts contain only counters and IDs. Prefer the
    typed result in the state delta so a later task can read the actual finding.
    """
    artifact = None
    if result.state_delta is not None:
        for name in (
            "paper_analysis_result", "qa_result", "preference_recommendation_result",
            "chart_result", "general_answer", "import_result", "parsed_document",
            "compressed_context_summary", "report",
        ):
            value = getattr(result.state_delta, name)
            if value is None:
                continue
            if isinstance(value, str):
                artifact = {"answer": value}
            elif isinstance(value, dict):
                artifact = deepcopy(value)
            elif hasattr(value, "model_dump"):
                artifact = value.model_dump(mode="json")
            if artifact is not None:
                break
    if artifact is None:
        artifact = deepcopy(result.metadata)
    answer_source = artifact.get("qa") if isinstance(artifact.get("qa"), dict) else artifact
    preview = next((
        answer_source[key] for key in ("answer", "summary", "markdown")
        if isinstance(answer_source.get(key), str) and answer_source[key].strip()
    ), result.observation)
    return artifact, preview[:1200]
