from evaluation.metrics import retrieval_recall_at_k
from evaluation.runner import aggregate_metrics, core_6_metrics
from evaluation.schemas import CaseMetricResult


def test_aggregate_metrics_builds_core_6_summary_for_custom_recall_k() -> None:
    results = [
        CaseMetricResult(
            case_id="case_1",
            kind="ask_document",
            actual_route="ask_document",
            task_success=True,
            keyword_recall=1.0,
            hit_at_k=True,
            recall_at_k=0.5,
            groundedness=0.75,
            route_correct=True,
            tool_call_success_count=2,
            tool_call_total=2,
            latency_ms=10,
        ),
        CaseMetricResult(
            case_id="case_2",
            kind="ask_document",
            actual_route="ask_document",
            task_success=False,
            keyword_recall=0.0,
            hit_at_k=False,
            recall_at_k=0.0,
            groundedness=0.25,
            route_correct=False,
            tool_call_success_count=1,
            tool_call_total=2,
            latency_ms=30,
        ),
    ]

    metrics = aggregate_metrics(results, recall_k=3)
    summary = core_6_metrics(metrics=metrics, recall_k=3)

    assert metrics.total_cases == 2
    assert metrics.task_success_rate == 0.5
    assert metrics.answer_keyword_recall == 0.5
    assert metrics.hit_at_k == 0.5
    assert metrics.recall_at_k == 0.25
    assert metrics.recall_at_5 is None
    assert summary.recall_k == 3
    assert summary.recall_at_k == 0.25
    assert summary.groundedness == 0.5
    assert summary.answer_keyword_recall == 0.5
    assert summary.route_accuracy == 0.5
    assert summary.tool_call_success_rate == 0.75
    assert summary.latency_p50_ms == 10
    assert summary.latency_p95_ms == 30


def test_aggregate_metrics_preserves_legacy_recall_at_5_fields() -> None:
    result = CaseMetricResult(
        case_id="case_1",
        kind="ask_document",
        actual_route="ask_document",
        task_success=True,
        hit_at_k=True,
        recall_at_k=1.0,
    )

    metrics = aggregate_metrics([result], recall_k=5)

    assert metrics.hit_at_5 == 1.0
    assert metrics.recall_at_5 == 1.0


def test_retrieval_recall_at_k_matches_source_and_document_ids() -> None:
    hit, recall, matched_evidence_ids, matched_source_ids, matched_keywords = retrieval_recall_at_k(
        expected_evidence_ids=[],
        expected_source_ids=["doc_a", "block_b"],
        expected_retrieval_keywords=[],
        hit_payloads=[
            {
                "id": "emb_1",
                "source_id": "block_a",
                "document_id": "doc_a",
                "content": "irrelevant",
                "evidence_ids": [],
            },
            {
                "id": "emb_2",
                "source_id": "block_b",
                "document_id": "doc_b",
                "content": "irrelevant",
                "evidence_ids": [],
            },
        ],
        k=2,
    )

    assert hit is True
    assert recall == 1.0
    assert matched_evidence_ids == []
    assert matched_source_ids == ["doc_a", "block_b"]
    assert matched_keywords == []
