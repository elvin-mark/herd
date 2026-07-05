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
