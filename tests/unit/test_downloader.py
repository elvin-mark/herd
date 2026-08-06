import os
import tempfile

import pytest

from herd.services.downloader import parse_model_identifier, resolve_model_path


def test_parse_model_identifier_valid():
    """Verifies parsing valid author/repo:tag and author/repo strings."""
    author, repo, tag = parse_model_identifier("unsloth/Qwen2.5-7B-GGUF:Q4_K_M")
    assert author == "unsloth"
    assert repo == "Qwen2.5-7B-GGUF"
    assert tag == "Q4_K_M"

    author2, repo2, tag2 = parse_model_identifier("unsloth/Llama-3.1-8B-GGUF")
    assert author2 == "unsloth"
    assert repo2 == "Llama-3.1-8B-GGUF"
    assert tag2 is None


def test_parse_model_identifier_invalid():
    """Verifies that malformed identifiers raise ValueError."""
    with pytest.raises(ValueError):
        parse_model_identifier("invalid_string_no_slash")


def test_resolve_model_path_local_absolute():
    """Verifies that existing local file paths are returned as absolute paths."""
    with tempfile.NamedTemporaryFile("w+", suffix=".gguf", delete=False) as f:
        f.write("dummy gguf bytes")
        path = f.name

    try:
        resolved = resolve_model_path(path)
        assert os.path.isabs(resolved)
        assert resolved == os.path.abspath(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_resolve_model_path_mmproj_exclusion(monkeypatch):
    """Verifies that resolve_model_path prioritizes non-mmproj files when resolving tags."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = os.path.join(tmp_dir, "models")
        repo_dir = os.path.join(models_dir, "huggingface", "author", "repo")
        os.makedirs(repo_dir, exist_ok=True)

        # Create both mmproj file and standard weight file matching tag 'Q4_K_M'
        mmproj_file = os.path.join(repo_dir, "mmproj-model-Q4_K_M.gguf")
        weight_file = os.path.join(repo_dir, "model-Q4_K_M.gguf")

        open(mmproj_file, "w").close()
        open(weight_file, "w").close()

        # Monkeypatch HERD_MODELS_DIR
        monkeypatch.setattr("herd.services.downloader.HERD_MODELS_DIR", models_dir)

        resolved = resolve_model_path("author/repo:Q4_K_M")
        assert resolved == weight_file
        assert "mmproj" not in os.path.basename(resolved)
