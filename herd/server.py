import os
import re
import time
import json
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

from herd.config import HERD_MODELS_DIR, HERD_PORT
from herd.manager import ProcessManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("herd.server")

# Global process manager
manager = ProcessManager()
cleanup_task = None


def list_downloaded_models() -> list[str]:
    """Scans the HERD_HOME/models/huggingface directory to find all downloaded GGUF/bin models."""
    models = []
    hf_dir = os.path.join(HERD_MODELS_DIR, "huggingface")
    if not os.path.exists(hf_dir):
        return models

    for author in os.listdir(hf_dir):
        author_path = os.path.join(hf_dir, author)
        if not os.path.isdir(author_path):
            continue
        for repo in os.listdir(author_path):
            repo_path = os.path.join(author_path, repo)
            if not os.path.isdir(repo_path):
                continue

            # Scan for .gguf or .bin files
            for root, _, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(".gguf") or file.endswith(".bin"):
                        name_no_ext, _ = os.path.splitext(file)

                        # Extract tag if we can match quantization format (e.g. Q4_K_M)
                        match = re.search(
                            r"([qI]?[0-9]_[A-Za-z0-9_]+)", name_no_ext, re.IGNORECASE
                        )
                        if match:
                            tag = match.group(1)
                        else:
                            tag = name_no_ext

                        models.append(f"{author}/{repo}:{tag}")

    return list(sorted(set(models)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global cleanup_task
    cleanup_task = asyncio.create_task(manager.cleanup_loop())
    logger.info(
        f"Herd Gateway Server listening on port {HERD_PORT}. Idle timeout cleanup task running."
    )
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
    for model_name in running_models:
        await manager.stop_model(model_name)
    logger.info("Herd Gateway Server stopped. All child processes terminated.")


app = FastAPI(title="Herd API Gateway", lifespan=lifespan)


async def proxy_to_port(
    port: int, path: str, request: Request, body_bytes: bytes
) -> Response:
    """Proxies an HTTP request to the target port, supporting streaming response."""
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

    try:
        response = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        logger.error(f"Failed to connect to backend server on port {port}: {e}")
        return JSONResponse(
            status_code=502,
            content={"error": f"Failed to connect to model server: {e}"},
        )

    async def response_generator():
        try:
            async for chunk in response.aiter_bytes():
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    res_headers = dict(response.headers)
    res_headers.pop("transfer-encoding", None)
    res_headers.pop("content-length", None)

    return StreamingResponse(
        response_generator(), status_code=response.status_code, headers=res_headers
    )


async def whisper_proxy(
    port: int,
    file: UploadFile,
    language: str | None,
    temperature: float | None,
    response_format: str | None,
    translate: bool,
) -> Response:
    """Proxies transcription/translation request to whisper-server's /inference endpoint."""
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

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            response = await client.post(
                f"http://127.0.0.1:{port}/inference", files=files, data=data
            )
        except Exception as e:
            logger.error(f"Failed to connect to whisper server on port {port}: {e}")
            return JSONResponse(
                status_code=502,
                content={"error": f"Failed to connect to whisper server: {e}"},
            )

        if response.status_code != 200:
            logger.error(
                f"Whisper server returned status {response.status_code}: {response.text}"
            )
            return Response(
                content=response.content,
                status_code=response.status_code,
                media_type="application/json",
            )

        result = response.json()

    text = result.get("text", "").strip()

    if response_format == "text":
        return Response(content=text, media_type="text/plain")
    elif response_format == "verbose_json":
        return JSONResponse(
            content={
                "text": text,
                "task": "translate" if translate else "transcribe",
                "language": result.get("language", language or "english"),
                "duration": result.get("duration", 0.0),
                "segments": result.get("segments", []),
            }
        )
    else:
        # Default or json
        return JSONResponse(content={"text": text})


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/v1/models")
async def get_models():
    """Lists all downloaded models in the HF repository style directory."""
    try:
        models = list_downloaded_models()
        data = []
        for model in models:
            data.append(
                {
                    "id": model,
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
    """Lists currently running model servers and their details."""
    active = []
    async with manager.lock:
        for name, info in manager.running_models.items():
            active.append(
                {
                    "model": name,
                    "port": info["port"],
                    "is_whisper": info["is_whisper"],
                    "is_embedding": info["is_embedding"],
                    "last_accessed": info["last_accessed"],
                    "idle_seconds": int(time.time() - info["last_accessed"]),
                    "log_path": info["log_path"],
                }
            )
    return active


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
            status_code=500, content={"error": f"Failed to start model server: {e}"}
        )


@app.post("/v1/models/unload")
async def unload_model(request: Request):
    """Explicitly stops a running model server process."""
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
            status_code=500, content={"error": f"Failed to start model server: {e}"}
        )

    return await proxy_to_port(port, "/v1/chat/completions", request, body_bytes)


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
            status_code=500, content={"error": f"Failed to start model server: {e}"}
        )

    return await proxy_to_port(port, "/v1/completions", request, body_bytes)


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
            status_code=500, content={"error": f"Failed to start model server: {e}"}
        )

    return await proxy_to_port(port, "/v1/embeddings", request, body_bytes)


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
            status_code=500, content={"error": f"Failed to start whisper server: {e}"}
        )

    return await whisper_proxy(
        port, file, language, temperature, response_format, translate=False
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
            status_code=500, content={"error": f"Failed to start whisper server: {e}"}
        )

    return await whisper_proxy(
        port, file, language, temperature, response_format, translate=True
    )
