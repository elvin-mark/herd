from fastapi.testclient import TestClient

from herd.api.server import app

client = TestClient(app)


def test_invalid_json_body():
    """Verifies that malformed JSON payloads return custom validation exceptions (400 Bad Request)."""
    response = client.post("/v1/chat/completions", content="invalid json")
    assert response.status_code == 400
    assert "error" in response.json()


def test_nonexistent_endpoint_404():
    """Verifies that non-existent API routes return 404 Not Found."""
    response = client.get("/v1/nonexistent_route")
    assert response.status_code == 404
