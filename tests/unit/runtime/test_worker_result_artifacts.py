from domain.schemas.research_functions import AnalyzePapersFunctionOutput
from runtime.research.agent_protocol import ResearchStateDelta, ResearchToolResult
from runtime.research.worker_results import worker_result_artifact


def test_analysis_artifact_contains_actual_answer_not_only_metrics():
    analysis = AnalyzePapersFunctionOutput(answer="Substantive finding. " * 200)
    result = ResearchToolResult(status="succeeded", observation="analyzed 2 papers",
                                metadata={"paper_count": 2},
                                state_delta=ResearchStateDelta(paper_analysis_result=analysis))
    artifact, summary = worker_result_artifact(result)
    assert artifact["answer"] == analysis.answer
    assert summary.startswith("Substantive finding.")
    assert len(summary) <= 1200
    artifact["key_points"].append("private")
    assert analysis.key_points == []


def test_result_without_delta_has_detached_payload_and_observation_summary():
    result = ResearchToolResult(status="failed", observation="provider unavailable",
                                metadata={"reason": "timeout", "details": []})
    artifact, summary = worker_result_artifact(result)
    assert summary == "provider unavailable"
    artifact["details"].append("private")
    assert result.metadata["details"] == []
