import os
import tempfile

from herd.services.rag import VectorDatabase, chunk_text, cosine_similarity


def test_rag_chunking():
    """Verifies that chunk_text splits long documents into overlapping slices."""
    text = "Word " * 500
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1


def test_rag_cosine_similarity():
    """Verifies cosine similarity calculation."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]

    assert abs(cosine_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(cosine_similarity(v1, v3) - 0.0) < 1e-5


def test_rag_database_operations():
    """Verifies that VectorDatabase handles SQLite chunk storage and retrieval cleanly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_rag.db")
        db = VectorDatabase(db_path=db_path)

        assert os.path.exists(db_path)

        # Insert test vector chunk
        db.insert_chunk(
            file_path="/tmp/test.py",
            chunk_index=0,
            text="def test(): pass",
            embedding=[0.1, 0.2, 0.3],
            model_name="test-embed-model",
        )

        indexed = db.list_indexed()
        assert len(indexed) == 1
        assert indexed[0][0] == "/tmp/test.py"

        # Cleanup
        deleted = db.delete_path("/tmp/test.py")
        assert deleted == 1
        assert len(db.list_indexed()) == 0
