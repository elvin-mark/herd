import os
import time
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import httpx

from herd.core.config import HERD_MODELS_DIR
from herd.services.manager import ProcessManager
from herd.core.metrics import collector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("herd.server")

# Global process manager
manager = ProcessManager()
cleanup_task = None


def list_downloaded_models() -> list[str]:
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: start process manager idle cleanup background loop
    global cleanup_task
    cleanup_task = asyncio.create_task(manager.cleanup_loop())
    logger.info("Herd Gateway Server started. Idle cleanup task scheduled.")
    yield
    # Shutdown
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    # Stop all models
    running_models = list(manager.running_models.keys())
    for model_path in running_models:
        # Since running_models is keyed by path, stop_model handles either model_name or path.
        # We can extract model_name from info dict
        info = manager.running_models.get(model_path)
        if info:
            await manager.stop_model(info["model_name"])
    logger.info("Herd Gateway Server stopped. All child processes terminated.")


app = FastAPI(title="Herd API Gateway", lifespan=lifespan)

# Mount assets folder to serve the logo image
assets_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "assets")
)
if os.path.exists(assets_dir):
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
@app.get("/dashboard")
async def get_dashboard():
    """Serves the Herd Web Control Center dashboard."""
    dashboard_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "templates", "dashboard.html")
    )
    try:
        with open(dashboard_path, "r", encoding="utf-8") as f:
            content = f.read()
        return Response(content=content, media_type="text/html")
    except Exception as e:
        return Response(
            content=f"Error loading dashboard: {e}",
            status_code=500,
            media_type="text/plain",
        )


@app.get("/metrics")
async def get_prometheus_metrics_endpoint():
    """Prometheus metrics scraping endpoint."""
    active = []
    async with manager.lock:
        for path, info in manager.running_models.items():
            name = info["model_name"]
            pid = info["process"].pid
            resources = manager.get_process_resources(pid)
            active.append(
                {
                    "model": name,
                    "port": info["port"],
                    "cpu_percent": resources["cpu_percent"],
                    "memory_bytes": resources["memory_bytes"],
                }
            )
    prometheus_data = collector.get_prometheus_metrics(active)
    return Response(
        content=prometheus_data,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


async def proxy_to_port(
    port: int, path: str, request: Request, body_bytes: bytes, model_name: str
) -> Response:
    """Proxies an HTTP request to the target port, supporting streaming response and recording metrics."""
    client = httpx.AsyncClient(timeout=None)
    url = f"http://127.0.0.1:{port}{path}"

    headers = dict(request.headers)
    headers["host"] = f"127.0.0.1:{port}"
    headers.pop("connection", None)
    headers.pop("content-length", None)

    req = client.build_request(
        method=request.method,
        url=url,
        headers=headers,
        content=body_bytes,
        params=request.query_params,
    )

    start_time = time.time()
    try:
        response = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        logger.error(f"Failed to connect to backend server on port {port}: {e}")
        collector.record_request(
            model_name=model_name,
            endpoint=path,
            duration_sec=time.time() - start_time,
            is_error=True,
        )
        return JSONResponse(
            status_code=502,
            content={"error": f"Failed to connect to model server: {e}"},
        )

    async def response_generator():
        full_response_text = b""
        try:
            async for chunk in response.aiter_bytes():
                full_response_text += chunk
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

            # Record request metrics at the end of streaming/non-streaming transmission
            duration = time.time() - start_time
            prompt_tokens = 0
            completion_tokens = 0

            try:
                # Extract prompt_tokens and completion_tokens using fast regex matching on returned payload
                import re

                text = full_response_text.decode("utf-8", errors="ignore")
                match_p = re.search(r'"prompt_tokens"\s*:\s*(\d+)', text)
                if match_p:
                    prompt_tokens = int(match_p.group(1))
                match_c = re.search(r'"completion_tokens"\s*:\s*(\d+)', text)
                if match_c:
                    completion_tokens = int(match_c.group(1))
            except Exception as e:
                logger.error(f"Error parsing tokens from response text: {e}")

            collector.record_request(
                model_name=model_name,
                endpoint=path,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_sec=duration,
                is_error=(response.status_code >= 400),
            )

    res_headers = dict(response.headers)
    res_headers.pop("transfer-encoding", None)
    res_headers.pop("content-length", None)

    return StreamingResponse(
        response_generator(),
        status_code=response.status_code,
        headers=res_headers,
    )


async def whisper_proxy(
    port: int,
    file: UploadFile,
    language: str | None,
    temperature: float | None,
    response_format: str | None,
    translate: bool,
    model: str,
) -> Response:
    """Proxies transcription/translation request to whisper-server's /inference endpoint and logs metrics."""
    file_bytes = await file.read()

    files = {
        "file": (
            file.filename or "audio.wav",
            file_bytes,
            file.content_type or "audio/wav",
        )
    }

    data = {}
    if language:
        data["language"] = language
    else:
        data["language"] = "auto"

    if temperature is not None:
        data["temperature"] = str(temperature)

    if translate:
        data["translate"] = "true"

    start_time = time.time()
    endpoint_path = (
        "/v1/audio/translations" if translate else "/v1/audio/transcriptions"
    )

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(
                f"http://127.0.0.1:{port}/inference", files=files, data=data
            )
        except Exception as e:
            logger.error(f"Failed to connect to whisper server on port {port}: {e}")
            collector.record_request(
                model_name=model,
                endpoint=endpoint_path,
                duration_sec=time.time() - start_time,
                is_error=True,
            )
            return JSONResponse(
                status_code=502,
                content={"error": f"Failed to connect to whisper server: {e}"},
            )

        duration = time.time() - start_time
        if response.status_code != 200:
            logger.error(
                f"Whisper server returned status {response.status_code}: {response.text}"
            )
            collector.record_request(
                model_name=model,
                endpoint=endpoint_path,
                duration_sec=duration,
                is_error=True,
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type="application/json",
            )

        result = response.json()

    text = result.get("text", "").strip()

    # Estimate word count as approximation for completion tokens (Whisper doesn't have prompt tokens in OpenAI sense)
    words_count = len(text.split())
    collector.record_request(
        model_name=model,
        endpoint=endpoint_path,
        prompt_tokens=0,
        completion_tokens=words_count,
        duration_sec=duration,
        is_error=False,
    )

    if response_format == "text":
        return Response(content=text, media_type="text/plain")

    openai_response = {
        "text": text,
        "segments": result.get("segments", []),
    }
    return JSONResponse(content=openai_response)


@app.get("/v1/models")
async def list_models():
    """Exposes downloaded models in OpenAI format."""
    try:
        models = list_downloaded_models()
        data = []
        for m in models:
            data.append(
                {
                    "id": m,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "huggingface",
                }
            )
        return {"object": "list", "data": data}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {"object": "list", "data": []}


@app.get("/v1/models/active")
async def get_active_models():
    """Lists currently running model servers and their resource/CPU details."""
    active = []
    async with manager.lock:
        for path, info in manager.running_models.items():
            name = info["model_name"]
            pid = info["process"].pid
            resources = manager.get_process_resources(pid)

            # Formatted memory size
            mem_bytes = resources["memory_bytes"]
            mem_gb = mem_bytes / (1024 * 1024 * 1024)
            mem_str = (
                f"{mem_gb:.2f} GB"
                if mem_gb >= 1.0
                else f"{mem_bytes / (1024 * 1024):.2f} MB"
            )

            active.append(
                {
                    "model": name,
                    "port": info["port"],
                    "is_whisper": info["is_whisper"],
                    "is_embedding": info["is_embedding"],
                    "last_accessed": info["last_accessed"],
                    "idle_seconds": int(time.time() - info["last_accessed"]),
                    "log_path": info["log_path"],
                    "cpu_percent": resources["cpu_percent"],
                    "memory_bytes": mem_bytes,
                    "memory_str": mem_str,
                }
            )
    return active


@app.get("/v1/models/stats")
async def get_model_stats():
    """Returns cumulative request, token, and performance stats for all models."""
    return collector.get_stats()


@app.post("/v1/models/load")
async def load_model(request: Request):
    """Explicitly starts a model server process."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    model_name = body.get("model")
    is_whisper = body.get("is_whisper", False)
    is_embedding = body.get("is_embedding", False)
    idle_timeout = body.get("idle_timeout")

    if not model_name:
        return JSONResponse(status_code=400, content={"error": "Missing 'model' field"})

    try:
        port = await manager.get_or_start_server(
            model_name,
            is_whisper=is_whisper,
            is_embedding=is_embedding,
            idle_timeout=idle_timeout,
        )
        return {"status": "loaded", "port": port}
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start model server: {e}"},
        )


@app.post("/v1/models/unload")
async def unload_model(request: Request):
    """Explicitly stops a model server process."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    model_name = body.get("model")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "Missing 'model' field"})

    await manager.stop_model(model_name)
    return {"status": "unloaded"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    model_name = body.get("model")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "Missing 'model' field"})

    try:
        port = await manager.get_or_start_server(
            model_name, is_whisper=False, is_embedding=False
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start model server: {e}"},
        )

    return await proxy_to_port(
        port, "/v1/chat/completions", request, body_bytes, model_name
    )


@app.post("/v1/completions")
async def completions(request: Request):
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    model_name = body.get("model")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "Missing 'model' field"})

    try:
        port = await manager.get_or_start_server(
            model_name, is_whisper=False, is_embedding=False
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start model server: {e}"},
        )

    return await proxy_to_port(
        port, "/v1/completions", request, body_bytes, model_name
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    model_name = body.get("model")
    if not model_name:
        return JSONResponse(status_code=400, content={"error": "Missing 'model' field"})

    try:
        port = await manager.get_or_start_server(
            model_name, is_whisper=False, is_embedding=True
        )
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start model server: {e}"},
        )

    return await proxy_to_port(
        port, "/v1/embeddings", request, body_bytes, model_name
    )


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile,
    model: str = Form(...),
    language: str | None = Form(None),
    temperature: float | None = Form(None),
    response_format: str | None = Form(None),
):
    try:
        port = await manager.get_or_start_server(model, is_whisper=True)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start whisper server: {e}"},
        )

    return await whisper_proxy(
        port,
        file,
        language,
        temperature,
        response_format,
        translate=False,
        model=model,
    )


@app.post("/v1/audio/translations")
async def audio_translations(
    file: UploadFile,
    model: str = Form(...),
    language: str | None = Form(None),
    temperature: float | None = Form(None),
    response_format: str | None = Form(None),
):
    try:
        port = await manager.get_or_start_server(model, is_whisper=True)
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to start whisper server: {e}"},
        )

    return await whisper_proxy(
        port,
        file,
        language,
        temperature,
        response_format,
        translate=True,
        model=model,
    )
