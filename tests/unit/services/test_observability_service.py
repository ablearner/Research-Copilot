import json

from services.research.observability_service import ResearchObservabilityService


def test_observability_service_writes_metrics_and_failures(tmp_path) -> None:
    service = ResearchObservabilityService(tmp_path / "observability")

    service.record_metric(metric_type="task_completed", payload={"task_id": "task-1"})
    service.archive_failure(failure_type="job_failed", payload={"job_id": "job-1"})

    metrics_lines = (tmp_path / "observability" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    failures_lines = (tmp_path / "observability" / "failures.jsonl").read_text(encoding="utf-8").splitlines()

    assert json.loads(metrics_lines[0])["metric_type"] == "task_completed"
    assert json.loads(failures_lines[0])["failure_type"] == "job_failed"
