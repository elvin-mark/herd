from fastapi.testclient import TestClient

from herd.api.server import app

client = TestClient(app)


def test_health_route():
    """Verifies that the gateway health route responds correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_history_route():
    """Verifies that the /v1/history endpoint returns recent request history."""
    response = client.get("/v1/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_pool_route():
    """Verifies that /v1/models/pool endpoint returns pool models and status."""
    response = client.get("/v1/models/pool")
    assert response.status_code == 200
    res = response.json()
    assert "pool" in res
    assert "status" in res
