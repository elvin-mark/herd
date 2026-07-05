import os
import logging
from typing import List
from fastapi import APIRouter, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from herd.core.config import HERD_MODELS_DIR
from herd.api.state import manager, pull_tasks
from herd.api.exceptions import HerdError
from herd.core.metrics import collector
from herd.core.utils import get_async_http_client

logger = logging.getLogger("herd.server.models")
router = APIRouter()


def list_downloaded_models() -> List[str]:
    """Scans the models directory and returns all model names in author/repo format."""
    models = []
    hf_dir = os.path.join(HERD_MODELS_DIR, "huggingface")
    if not os.path.exists(hf_dir):
        return models

    # Walk directory to find author/repo structures containing .gguf or .bin
    for author in os.listdir(hf_dir):
        author_path = os.path.join(hf_dir, author)
        if not os.path.isdir(author_path):
            continue

        for repo in os.listdir(author_path):
            repo_path = os.path.join(author_path, repo)
            if not os.path.isdir(repo_path):
                continue

            # Look for model files
            files = os.listdir(repo_path)
            model_files = [
                f for f in files if f.endswith(".gguf") or f.endswith(".bin")
            ]
            if model_files:
                models.append(f"{author}/{repo}")

    return sorted(models)


@router.get("/v1/models")
async def list_models():
    """Lists all downloaded local models."""
    local_models = list_downloaded_models()
    data = []
    for m in local_models:
        data.append({"id": m, "object": "model", "owned_by": "user"})
    return {"data": data}


@router.get("/v1/models/active")
async def list_active_models():
    """Lists currently running model servers."""
    active = []
    async with manager.lock:
        for path, info in manager.running_models.items():
            active.append(
                {
                    "model": info["model_name"],
                    "port": info["port"],
                    "is_whisper": info.get("is_whisper", False),
                    "is_embedding": info.get("is_embedding", False),
                    "pid": info["process"].pid,
                    "memory_bytes": manager.get_process_resources(
                        info["process"].pid
                    )["memory_bytes"],
                }
            )
    return active


@router.get("/v1/models/stats")
async def get_stats_endpoint():
    """Returns cumulative request, token, and performance stats for all models."""
    return collector.get_stats()


@router.post("/v1/models/load")
async def load_model(request: Request):
    """Explicitly starts a model server process."""
    try:
        body = await request.json()
    except Exception:
        raise HerdError("Invalid JSON body", status_code=400)

    model_name = body.get("model")
    is_whisper = body.get("is_whisper", False)
    is_embedding = body.get("is_embedding", False)
    idle_timeout = body.get("idle_timeout")

    if not model_name:
        raise HerdError("Missing 'model' field", status_code=400)

    # Check if this model targets a registered cloud provider
    if ":" in model_name:
        from herd.core.config import settings

        settings.reload()
        parts = model_name.split(":", 1)
        if parts[0] in settings.providers:
            return {"status": "loaded", "port": 0, "provider": parts[0]}

    try:
        port = await manager.get_or_start_server(
            model_name,
            is_whisper=is_whisper,
            is_embedding=is_embedding,
            idle_timeout=idle_timeout,
        )
        return {"status": "loaded", "port": port}
    except FileNotFoundError as e:
        raise HerdError(str(e), status_code=404)


@router.post("/v1/models/unload")
async def unload_model(request: Request):
    """Explicitly stops a model server process."""
    try:
        body = await request.json()
    except Exception:
        raise HerdError("Invalid JSON body", status_code=400)

    model_name = body.get("model")
    if not model_name:
        raise HerdError("Missing 'model' field", status_code=400)

    # Check if this model targets a registered cloud provider
    if ":" in model_name:
        from herd.core.config import settings

        settings.reload()
        parts = model_name.split(":", 1)
        if parts[0] in settings.providers:
            return {"status": "unloaded"}

    await manager.stop_model(model_name)
    return {"status": "unloaded"}


@router.get("/v1/hf/search")
async def hf_search(query: str, limit: int = 10):
    """Searches Hugging Face Hub for GGUF models."""
    url = f"https://huggingface.co/api/models?search={query}&filter=gguf&sort=downloads&direction=-1&limit={limit}"
    client = get_async_http_client()
    try:
        res = await client.get(url, timeout=10.0)
        if res.status_code != 200:
            return JSONResponse(
                status_code=res.status_code, content={"error": res.text}
            )
        return res.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/v1/models/pull")
async def pull_model(request: Request, background_tasks: BackgroundTasks):
    """Starts pulling a model in the background and tracks progress."""
    body = await request.json()
    model_name = body.get("model")
    if not model_name:
        return JSONResponse(
            status_code=400, content={"error": "Missing 'model' field"}
        )

    if model_name in pull_tasks and pull_tasks[model_name]["status"] in [
        "downloading",
        "pending",
    ]:
        return {"status": "already_pulling"}

    pull_tasks[model_name] = {"status": "pending", "progress": 0, "error": None}

    async def download_worker(name: str):
        from herd.services.downloader import (
            parse_model_identifier,
            list_hf_repository_files,
        )

        try:
            author, repo, tag = parse_model_identifier(name)
            files = await list_hf_repository_files(author, repo)
            model_files = [
                f for f in files if f.endswith(".gguf") or f.endswith(".bin")
            ]
            if not model_files:
                pull_tasks[name] = {
                    "status": "failed",
                    "progress": 0,
                    "error": "No model files found",
                }
                return

            chosen_file = model_files[0]
            if tag:
                matches = [f for f in model_files if tag.lower() in f.lower()]
                if matches:
                    chosen_file = matches[0]

            download_url = (
                f"https://huggingface.co/{author}/{repo}/resolve/main/{chosen_file}"
            )
            dest_path = os.path.join(
                HERD_MODELS_DIR, "huggingface", author, repo, chosen_file
            )

            pull_tasks[name]["status"] = "downloading"

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            client = get_async_http_client()
            async with client.stream("GET", download_url) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(dest_path, "wb") as f:
                    async for chunk in r.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pull_tasks[name]["progress"] = int(
                                (downloaded / total) * 100
                            )

            pull_tasks[name]["status"] = "completed"
            pull_tasks[name]["progress"] = 100
        except Exception as e:
            pull_tasks[name] = {"status": "failed", "progress": 0, "error": str(e)}

    background_tasks.add_task(download_worker, model_name)
    return {"status": "started"}


@router.get("/v1/models/pull/status")
async def pull_status():
    """Gets status of all pulling tasks."""
    return pull_tasks
