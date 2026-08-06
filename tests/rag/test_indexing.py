import os
import tempfile

import pytest

from herd.services.rag import VectorDatabase, index_directory


@pytest.mark.anyio
async def test_index_directory_async(monkeypatch):
    """Verifies that index_directory scans text files, generates chunks, and writes chunks to .herd-index.db."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create sample files
        py_file = os.path.join(tmp_dir, "sample.py")
        md_file = os.path.join(tmp_dir, "README.md")

        with open(py_file, "w") as f:
            f.write("def sample_function():\n    return 'Hello World'\n")

        with open(md_file, "w") as f:
            f.write("# Project Documentation\nThis is a sample readme.\n")

        # Monkeypatch get_embedding to return dummy 3D float vector
        async def mock_get_embedding(text, model):
            return [0.1, 0.2, 0.3]

        monkeypatch.setattr("herd.services.rag.get_embedding", mock_get_embedding)

        chunks_count = await index_directory(tmp_dir, embedding_model="test-embed-model")
        assert chunks_count >= 2

        db_path = os.path.join(tmp_dir, ".herd-index.db")
        assert os.path.exists(db_path)

        db = VectorDatabase(db_path)
        indexed = db.list_indexed()
        assert len(indexed) >= 2
