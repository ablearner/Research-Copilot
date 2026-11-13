from benchmarks.backend.run import summarize


def test_latency_summary_uses_nearest_rank_and_keeps_errors():
    result = summarize([1, 2, 3, 4, 100], [0.1, 0.2], 2.0, ["failure"])
    assert result["p50_ms"] == 3
    assert result["p95_ms"] == 100
    assert result["p99_ms"] == 100
    assert result["successful_operations_per_second"] == 2.5
    assert result["errors"] == 1
    assert result["error_rate"] == 1 / 6
    assert result["latency_samples_ms"] == [1, 2, 3, 4, 100]
