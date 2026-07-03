import os
import sqlite3
import array
import httpx
from typing import List, Dict, Any
from herd.core.config import HERD_HOME, HERD_PORT

DB_PATH = os.path.join(HERD_HOME, "embeddings.db")


def init_db():
    """Initializes the SQLite vector database schema."""
    os.makedirs(HERD_HOME, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()


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
    """Calls the local Herd Gateway to generate embeddings for a given text."""
    url = f"http://127.0.0.1:{HERD_PORT}/v1/embeddings"
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json={"model": model_name, "input": text},
            timeout=30.0
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


async def index_directory(directory_path: str, embedding_model: str) -> int:
    """Recursively parses text-based files in a directory, chunks them, embeds them, and indexes in DB."""
    init_db()
    
    supported_extensions = {
        ".txt", ".md", ".py", ".js", ".ts", ".html", ".css",
        ".json", ".yaml", ".yml", ".sh", ".ini", ".cfg", ".sql"
    }

    chunks_added = 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for root, _, files in os.walk(directory_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() not in supported_extensions:
                continue

            file_path = os.path.abspath(os.path.join(root, file))
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                file_chunks = chunk_text(content)
                for idx, text_chunk in enumerate(file_chunks):
                    # Call embedding API
                    vector = await get_embedding(text_chunk, embedding_model)
                    # Convert float list to binary blob
                    vector_blob = array.array('f', vector).tobytes()

                    cursor.execute(
                        "INSERT INTO chunks (file_path, chunk_index, text, embedding, model_name) VALUES (?, ?, ?, ?, ?)",
                        (file_path, idx, text_chunk, vector_blob, embedding_model)
                    )
                    chunks_added += 1

            except Exception:
                # Silently skip files that fail to read/encode
                continue

    conn.commit()
    conn.close()
    return chunks_added


def search_vectors(query_vector: List[float], embedding_model: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs a semantic search by calculating cosine similarity over stored DB vectors."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query all indexed vectors for the given model
    cursor.execute(
        "SELECT file_path, text, embedding FROM chunks WHERE model_name = ?",
        (embedding_model,)
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for file_path, text, blob in rows:
        # Convert binary blob back to float list
        vector = list(array.array('f', blob))
        
        sim = cosine_similarity(query_vector, vector)
        results.append({
            "file_path": file_path,
            "text": text,
            "similarity": sim
        })

    # Sort results by similarity descending
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def list_indexed_files() -> List[tuple]:
    """Retrieves file path, embedding model, and chunk counts for all indexed items."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT file_path, model_name, COUNT(*) FROM chunks GROUP BY file_path, model_name"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def remove_indexed_path(target_path: str) -> int:
    """Deletes chunks matching the target path or located under the target directory."""
    init_db()
    abs_target = os.path.abspath(target_path)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Match the exact file path or anything under it as a sub-path
    cursor.execute(
        "DELETE FROM chunks WHERE file_path = ? OR file_path LIKE ?",
        (abs_target, abs_target + os.sep + "%")
    )
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted
