import os
import logging
from typing import Optional
from fastapi import APIRouter, Request, BackgroundTasks

from herd.api.state import manager
from herd.api.exceptions import HerdError
from herd.services.rag import (
    list_indexed_files,
    remove_indexed_path,
    index_directory,
    get_embedding,
    search_vectors,
)

logger = logging.getLogger("herd.server.db")
router = APIRouter()


@router.get("/v1/db/list")
async def db_list(directory: Optional[str] = None):
    """Lists indexed files in the local database (optionally targeting a specific directory)."""
    rows = list_indexed_files(directory)
    data = [{"file_path": r[0], "model_name": r[1], "chunks": r[2]} for r in rows]
    return data


@router.post("/v1/db/remove")
async def db_remove(request: Request):
    """Removes a file path from the vector database index."""
    body = await request.json()
    path = body.get("path")
    if not path:
        raise HerdError("Missing 'path' field", status_code=400)
    abs_path = os.path.abspath(path)
    count = remove_indexed_path(abs_path)
    return {"status": "removed", "count": count}


@router.post("/v1/db/index")
async def db_index(request: Request, background_tasks: BackgroundTasks):
    """Triggers RAG indexing on a local directory in the background."""
    body = await request.json()
    directory = body.get("directory")
    model_name = body.get("model")
    if not directory or not model_name:
        raise HerdError("Missing 'directory' or 'model' field", status_code=400)

    if not os.path.exists(directory):
        raise HerdError("Directory not found", status_code=404)

    await manager.get_or_start_server(model_name, is_whisper=False, is_embedding=True)

    async def index_worker(d: str, m: str):
        try:
            await index_directory(d, m)
        except Exception as e:
            logger.error(f"Background indexing failed: {e}")

    background_tasks.add_task(index_worker, directory, model_name)
    return {"status": "indexing_started"}


@router.post("/v1/db/search")
async def db_search(request: Request):
    """Performs semantic vector RAG search matching the user query."""
    body = await request.json()
    query = body.get("query")
    model_name = body.get("model")
    limit = body.get("limit", 5)
    directory = body.get("directory")

    if not query:
        raise HerdError("Missing 'query' field", status_code=400)

    if not model_name:
        from herd.services.rag import detect_db_embedding_model

        model_name = detect_db_embedding_model(directory)
        if not model_name:
            from herd.core.config import settings

            model_name = settings.default_embedding

    if not model_name:
        raise HerdError(
            "No embedding model specified and none could be auto-detected",
            status_code=400,
        )

    await manager.get_or_start_server(model_name, is_whisper=False, is_embedding=True)
    query_vector = await get_embedding(query, model_name)
    matches = search_vectors(
        query_vector, model_name, top_k=limit, target_path=directory
    )
    return matches
