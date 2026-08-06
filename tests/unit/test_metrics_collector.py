from herd.core.metrics import MetricsCollector


def test_metrics_collector_ring_buffer():
    """Verifies that MetricsCollector limits history entries to specified max_history."""
    collector = MetricsCollector(max_history=50, db_path=":memory:")
    for i in range(70):
        collector.record_request(
            model_name="test-model",
            endpoint="/v1/chat/completions",
            prompt_tokens=10,
            completion_tokens=20,
            duration_sec=0.5,
            prompt_snippet=f"test prompt {i}",
        )

    history = collector.get_history(limit=50)
    assert len(history) == 50
    assert history[0]["prompt_snippet"] == "test prompt 69"


def test_metrics_collector_in_flight_counter():
    """Verifies in-flight request counter increments and decrements."""
    collector = MetricsCollector(max_history=10, db_path=":memory:")
    assert collector.get_in_flight("model-a") == 0

    collector.inc_in_flight("model-a")
    collector.inc_in_flight("model-a")
    assert collector.get_in_flight("model-a") == 2

    collector.dec_in_flight("model-a")
    assert collector.get_in_flight("model-a") == 1

    collector.dec_in_flight("model-a")
    assert collector.get_in_flight("model-a") == 0
