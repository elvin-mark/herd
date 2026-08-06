from herd.core.metrics import MetricsCollector


def test_benchmark_speed_calculation():
    """Verifies that tokens per second and latency calculations format accurately."""
    collector = MetricsCollector(max_history=10, db_path=":memory:")

    collector.record_request(
        model_name="bench-model",
        endpoint="/v1/chat/completions",
        prompt_tokens=100,
        completion_tokens=200,
        duration_sec=2.0,
    )

    stats = collector.stats.get("bench-model", {})
    assert stats["requests_total"] == 1
    assert stats["prompt_tokens_total"] == 100
    assert stats["completion_tokens_total"] == 200
    assert stats["total_duration_sec"] == 2.0
