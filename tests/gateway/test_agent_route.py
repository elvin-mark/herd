from fastapi.testclient import TestClient

from herd.api.server import app

client = TestClient(app)


def test_agent_run_route_validation(monkeypatch):
    """Verifies POST /v1/agent/run route accepts objective payload."""

    async def mock_get_or_start(model_name, is_whisper=False, is_embedding=False):
        return 11434

    monkeypatch.setattr("herd.api.state.manager.get_or_start_server", mock_get_or_start)

    response = client.post(
        "/v1/agent/run",
        json={
            "objective": "Test agent objective",
            "model": "unsloth/Qwen2.5-7B-GGUF",
            "max_turns": 1,
        },
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
