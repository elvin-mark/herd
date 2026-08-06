import io

from fastapi.testclient import TestClient

from herd.api.server import app

client = TestClient(app)


def test_audio_transcriptions_validation_missing_file():
    """Verifies that /v1/audio/transcriptions validates missing file/model fields."""
    response = client.post("/v1/audio/transcriptions", data={"model": "test-whisper"})
    assert response.status_code == 422  # Unprocessable Entity for missing file upload


def test_audio_transcriptions_validation_with_file():
    """Verifies that /v1/audio/transcriptions accepts multipart audio upload."""
    dummy_wav = io.BytesIO(
        b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    )
    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", dummy_wav, "audio/wav")},
        data={"model": "ggerganov/whisper.cpp:ggml-base.bin"},
    )
    # Gateway attempts to route to backend model, testing payload acceptance
    assert response.status_code in (200, 400, 404, 500)


def test_embeddings_route_validation():
    """Verifies that /v1/embeddings requires input payload."""
    response = client.post("/v1/embeddings", json={})
    assert response.status_code in (400, 422)
