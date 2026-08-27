from fastapi.testclient import TestClient

from herd.api.server import app

client = TestClient(app)


def test_models_list_route():
    """Verifies GET /v1/models endpoint returns list object."""
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert "data" in data
    assert isinstance(data["data"], list)


def test_models_active_route():
    """Verifies GET /v1/models/active endpoint returns active running processes list."""
    res = client.get("/v1/models/active")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_metrics_prometheus_route():
    """Verifies GET /metrics Prometheus scraping route returns text/plain metrics format."""
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers.get("content-type", "")
    assert "herd_" in res.text


def test_ollama_tags_route():
    """Verifies GET /api/tags endpoint returns models array."""
    res = client.get("/api/tags")
    assert res.status_code == 200
    data = res.json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_ollama_version_route():
    """Verifies GET /api/version endpoint returns version info."""
    res = client.get("/api/version")
    assert res.status_code == 200
    assert "version" in res.json()
