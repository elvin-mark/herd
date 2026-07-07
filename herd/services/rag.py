import os
import sqlite3
import array
import asyncio
from typing import List, Dict, Any, Optional
from herd.core.config import HERD_HOME, HERD_PORT

DB_PATH = os.path.join(HERD_HOME, "embeddings.db")


class VectorDatabase:
    """Repository abstraction layer for SQLite vector database."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        """Initializes the SQLite database schema if not present."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    embedding BLOB,
                    model_name TEXT
                )
            """)
            conn.commit()

    def insert_chunk(
        self,
        file_path: str,
        chunk_index: int,
        text: str,
        embedding: List[float],
        model_name: str,
    ):
        """Inserts a single embedded document chunk into the database."""
        vector_blob = array.array("f", embedding).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chunks (file_path, chunk_index, text, embedding, model_name) VALUES (?, ?, ?, ?, ?)",
                (file_path, chunk_index, text, vector_blob, model_name),
            )
            conn.commit()

    def get_all_vectors(self, model_name: str) -> List[tuple]:
        """Retrieves all indexed vectors matching the specified model name (robust against tag omissions)."""
        # Find all unique model names stored in chunks table
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT model_name FROM chunks")
            distinct_names = [row[0] for row in cursor.fetchall()]

        # Determine which stored names are equivalent to requested model_name
        matching_names = []
        for name in distinct_names:
            if name == model_name:
                matching_names.append(name)
                continue

            # Try parsing both to compare author and repo
            try:
                from herd.services.downloader import parse_model_identifier

                a1, r1, _ = parse_model_identifier(name)
                a2, r2, _ = parse_model_identifier(model_name)
                if a1.lower() == a2.lower() and r1.lower() == r2.lower():
                    matching_names.append(name)
            except Exception:
                # Fallback to case-insensitive prefix/suffix matching
                if name.lower().startswith(
                    model_name.lower()
                ) or model_name.lower().startswith(name.lower()):
                    matching_names.append(name)

        if not matching_names:
            matching_names = [model_name]

        # Retrieve chunks matching any of the matching model names
        placeholders = ",".join(["?"] * len(matching_names))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT file_path, text, embedding FROM chunks WHERE model_name IN ({placeholders})",
                tuple(matching_names),
            )
            return cursor.fetchall()

    def list_indexed(self) -> List[tuple]:
        """Lists file paths, model configurations, and chunk counts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT file_path, model_name, COUNT(*) FROM chunks GROUP BY file_path, model_name"
            )
            return cursor.fetchall()

    def delete_path(self, target_path: str) -> int:
        """Deletes all chunks matching or located underneath a specific path."""
        abs_target = os.path.abspath(target_path)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM chunks WHERE file_path = ? OR file_path LIKE ?",
                (abs_target, abs_target + os.sep + "%"),
            )
            conn.commit()
            return cursor.rowcount


# Instantiate DB Repository singleton
db = VectorDatabase()


def init_db():
    """Initializes the database schema (kept for backward compatibility)."""
    db.init_db()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Splits a document text into overlapping sliding-window chunks."""
    chunks = []
    if not text:
        return chunks

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


async def get_embedding(text: str, model_name: str) -> List[float]:
    """Calls the local Herd Gateway using the pooled HTTP client to generate embeddings."""
    from herd.core.utils import get_async_http_client

    url = f"http://127.0.0.1:{HERD_PORT}/v1/embeddings"
    client = get_async_http_client()
    response = await client.post(
        url, json={"model": model_name, "input": text}, timeout=30.0
    )
    if response.status_code != 200:
        raise RuntimeError(f"Embedding request failed: {response.text}")

    result = response.json()
    return result["data"][0]["embedding"]


def dot_product(v1: List[float], v2: List[float]) -> float:
    return sum(x * y for x, y in zip(v1, v2))


def magnitude(v: List[float]) -> float:
    return sum(x * x for x in v) ** 0.5


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    mag1 = magnitude(v1)
    mag2 = magnitude(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product(v1, v2) / (mag1 * mag2)


async def _read_file_content_async(file_path: str) -> str:
    """Reads file contents asynchronously in a separate worker thread."""

    def _read():
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    return await asyncio.to_thread(_read)


async def index_directory(
    directory_path: str, embedding_model: str, types: Optional[str] = None
) -> int:
    """Recursively parses text-based files in a directory, chunks them, embeds them, and indexes in DB."""
    if types:
        supported_extensions = {
            ext.strip().lower() for ext in types.split(",") if ext.strip()
        }
    else:
        supported_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".ts",
            ".html",
            ".css",
            ".json",
            ".yaml",
            ".yml",
            ".sh",
            ".ini",
            ".cfg",
            ".sql",
        }

    chunks_added = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() not in supported_extensions:
                continue

            file_path = os.path.abspath(os.path.join(root, file))
            try:
                # Read content using non-blocking threaded file I/O
                content = await _read_file_content_async(file_path)

                file_chunks = chunk_text(content)
                for idx, text_chunk in enumerate(file_chunks):
                    # Call embedding API
                    vector = await get_embedding(text_chunk, embedding_model)
                    # Insert using DB repository
                    db.insert_chunk(file_path, idx, text_chunk, vector, embedding_model)
                    chunks_added += 1

            except Exception:
                # Silently skip files that fail to read or embed
                continue

    return chunks_added


def search_vectors(
    query_vector: List[float], embedding_model: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """Performs a semantic search by calculating cosine similarity over stored DB vectors."""
    rows = db.get_all_vectors(embedding_model)
    results = []
    for file_path, text, blob in rows:
        # Convert binary blob back to float list
        vector = list(array.array("f", blob))

        sim = cosine_similarity(query_vector, vector)
        results.append({"file_path": file_path, "text": text, "similarity": sim})

    # Sort results by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def list_indexed_files() -> List[tuple]:
    """Retrieves file path, embedding model, and chunk counts for all indexed items."""
    return db.list_indexed()


def remove_indexed_path(target_path: str) -> int:
    """Deletes chunks matching the target path or located under the target directory."""
    return db.delete_path(target_path)
