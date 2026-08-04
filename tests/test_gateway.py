from fastapi.testclient import TestClient

from herd.api.server import app

client = TestClient(app)


def test_health_route():
    """Verifies that the gateway health route responds correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_json_body():
    """Verifies that malformed JSON payloads return custom validation exceptions."""
    response = client.post("/v1/chat/completions", content="invalid json")
    assert response.status_code == 400
    assert "error" in response.json()


def test_history_route():
    """Verifies that the /v1/history endpoint returns recent request history."""
    response = client.get("/v1/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_metrics_collector_ring_buffer():
    """Verifies that MetricsCollector limits history entries to 50."""
    from herd.core.metrics import MetricsCollector

    collector = MetricsCollector(max_history=50)
    for i in range(70):
        collector.record_request(
            model_name="test-model",
            endpoint="/v1/chat/completions",
            prompt_tokens=10,
            completion_tokens=20,
            duration_sec=0.5,
            prompt_snippet=f"test prompt {i}",
        )

    history = collector.get_history()
    assert len(history) == 50
    assert history[0]["prompt_snippet"] == "test prompt 69"


def test_pool_route():
    """Verifies that /v1/models/pool endpoint returns pool models and status."""
    response = client.get("/v1/models/pool")
    assert response.status_code == 200
    res = response.json()
    assert "pool" in res
    assert "status" in res


def test_least_busy_load_balancer():
    """Verifies that ProcessManager selects the running model with the lowest in-flight request count."""
    from herd.services.manager import ProcessManager

    pm = ProcessManager()
    pm.running_models = {
        "/path/model1": {"model_name": "model1", "in_flight": 5},
        "/path/model2": {"model_name": "model2", "in_flight": 1},
        "/path/model3": {"model_name": "model3", "in_flight": 3},
    }

    selected = pm.select_least_busy_pool_model(["model1", "model2", "model3"])
    assert selected == "model2"
