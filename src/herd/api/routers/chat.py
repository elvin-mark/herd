import json
import logging
import time

from fastapi import APIRouter, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from herd.api.exceptions import HerdError
from herd.api.state import manager
from herd.core.metrics import collector
from herd.core.utils import get_async_http_client

logger = logging.getLogger("herd.server.chat")
router = APIRouter()


async def proxy_to_cloud(
    provider: str, target_model: str, request: Request, body: dict, path: str
) -> Response:
    """Proxies an HTTP request to a remote cloud provider, supporting streaming response and recording metrics."""
    from herd.core.config import settings

    prov_config = settings.providers.get(provider)
    if not prov_config:
        raise HerdError(
            f"Cloud provider '{provider}' is not configured in settings.",
            status_code=400,
        )

    api_key = prov_config.get("api_key")
    base_url = prov_config.get("base_url", "").rstrip("/")
    if not base_url:
        raise HerdError(f"Base URL is missing for provider '{provider}'.", status_code=400)

    # Rewrite model field to match the remote provider target model name
    body["model"] = target_model

    # Prevent duplicate '/v1/v1' path prefix nesting if the provider base URL already ends with '/v1'
    path_suffix = path
    if base_url.endswith("/v1") and path.startswith("/v1"):
        path_suffix = path[3:]  # Strip '/v1' (e.g. '/v1/chat/completions' -> '/chat/completions')

    url = f"{base_url}{path_suffix}"

    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    client = get_async_http_client()
    req = client.build_request(
        method="POST",
        url=url,
        headers=headers,
        json=body,
        params=request.query_params,
    )

    cloud_model_id = f"{provider}:{target_model}"
    collector.inc_in_flight(cloud_model_id)
    start_time = time.time()
    try:
        response = await client.send(req, stream=True)
    except Exception as e:
        collector.dec_in_flight(cloud_model_id)
        logger.error(f"Failed to connect to cloud provider '{provider}' at {url}: {e}")
        raise HerdError(f"Cloud provider connection failed: {e}", status_code=502)

    if response.status_code >= 400:
        collector.dec_in_flight(cloud_model_id)
        content = await response.aread()
        try:
            err_json = json.loads(content)
            err_msg = (
                err_json.get("error", {}).get("message") or err_json.get("error") or str(content)
            )
        except Exception:
            err_msg = content.decode("utf-8", errors="ignore")
        raise HerdError(
            f"Cloud provider '{provider}' returned error {response.status_code}: {err_msg}",
            status_code=response.status_code,
        )

    async def response_generator():
        full_response_text = b""
        try:
            async for chunk in response.aiter_bytes():
                full_response_text += chunk
                yield chunk
        finally:
            collector.dec_in_flight(cloud_model_id)
            await response.aclose()

            # Record request metrics at the end of streaming/non-streaming transmission
            duration = time.time() - start_time
            prompt_tokens = 0
            completion_tokens = 0

            try:
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

            prompt_snippet = ""
            full_prompt = body
            if isinstance(body, dict):
                if "prompt" in body:
                    prompt_snippet = str(body["prompt"])[:100]
                elif "messages" in body and isinstance(body["messages"], list):
                    last_user_msg = next(
                        (
                            m.get("content", "")
                            for m in reversed(body["messages"])
                            if isinstance(m, dict) and m.get("role") == "user"
                        ),
                        "",
                    )
                    prompt_snippet = str(last_user_msg)[:100]
                elif "input" in body:
                    prompt_snippet = str(body["input"])[:100]

            full_resp = None
            try:
                full_resp = json.loads(text)
            except Exception:
                full_resp = text

            response_snippet = ""
            if isinstance(full_resp, dict):
                choices = full_resp.get("choices", [])
                if choices and isinstance(choices, list):
                    msg = choices[0].get("message", {})
                    response_snippet = msg.get("content", "")[:100] if isinstance(msg, dict) else ""
                elif "data" in full_resp and isinstance(full_resp["data"], list):
                    response_snippet = f"Generated {len(full_resp['data'])} embedding vector(s)"
            elif isinstance(full_resp, str):
                response_snippet = full_resp[:100]

            # Record cloud metrics under provider:target_model identifier
            collector.record_request(
                model_name=cloud_model_id,
                endpoint=path_suffix,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_sec=duration,
                prompt_snippet=prompt_snippet,
                response_snippet=response_snippet,
                full_prompt=full_prompt,
                full_response=full_resp,
                is_error=False,
            )

    return StreamingResponse(
        response_generator(),
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


def resolve_pool_or_model(model_name: str) -> str:
    """Resolves 'auto' or 'pool' aliases strictly to the least-busy running model in the configured pool."""
    from herd.core.config import settings

    settings.reload()
    if model_name.lower() in ("auto", "default", "pool"):
        pool_models = settings.pool
        if pool_models:
            selected = collector.select_least_busy_pool_model(pool_models)
            if selected:
                logger.info(
                    f"Pool Load Balancer routed '{model_name}' request to pool instance '{selected}'."
                )
                return selected

        if settings.default_llm:
            return settings.default_llm

        raise HerdError(
            "Your load balancing pool is currently empty. Add models using 'herd pool add <model_name>'.",
            status_code=400,
        )
    return model_name


async def proxy_to_port(
    port: int, path: str, request: Request, body_bytes: bytes, model_name: str
) -> Response:
    """Proxies an HTTP request to the target port, supporting streaming response and recording metrics."""
    client = get_async_http_client()
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

    manager.inc_in_flight(model_name)
    start_time = time.time()
    try:
        response = await client.send(req, stream=True)
    except Exception as e:
        manager.dec_in_flight(model_name)
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
            manager.dec_in_flight(model_name)
            await response.aclose()

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

            prompt_snippet = ""
            full_prompt = None
            if body_bytes:
                try:
                    full_prompt = json.loads(body_bytes)
                    if isinstance(full_prompt, dict):
                        if "prompt" in full_prompt:
                            prompt_snippet = str(full_prompt["prompt"])[:100]
                        elif "messages" in full_prompt and isinstance(
                            full_prompt["messages"], list
                        ):
                            last_user_msg = next(
                                (
                                    m.get("content", "")
                                    for m in reversed(full_prompt["messages"])
                                    if isinstance(m, dict) and m.get("role") == "user"
                                ),
                                "",
                            )
                            prompt_snippet = str(last_user_msg)[:100]
                except Exception:
                    full_prompt = body_bytes.decode("utf-8", errors="ignore")
                    prompt_snippet = full_prompt[:100]

            full_resp = None
            try:
                full_resp = json.loads(text)
            except Exception:
                full_resp = text

            response_snippet = ""
            if isinstance(full_resp, dict):
                choices = full_resp.get("choices", [])
                if choices and isinstance(choices, list):
                    msg = choices[0].get("message", {})
                    response_snippet = msg.get("content", "")[:100] if isinstance(msg, dict) else ""
            elif isinstance(full_resp, str):
                response_snippet = full_resp[:100]

            collector.record_request(
                model_name=model_name,
                endpoint=path,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                duration_sec=duration,
                is_error=(response.status_code >= 400),
                prompt_snippet=prompt_snippet,
                response_snippet=response_snippet,
                full_prompt=full_prompt,
                full_response=full_resp,
            )

    res_headers = dict(response.headers)
    res_headers.pop("transfer-encoding", None)
    res_headers.pop("content-length", None)
    media_type = res_headers.get("content-type")

    return StreamingResponse(
        response_generator(),
        status_code=response.status_code,
        headers=res_headers,
        media_type=media_type,
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
    endpoint_path = "/v1/audio/translations" if translate else "/v1/audio/transcriptions"

    client = get_async_http_client()
    try:
        response = await client.post(f"http://127.0.0.1:{port}/inference", files=files, data=data)
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
        logger.error(f"Whisper server returned status {response.status_code}: {response.text}")
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


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except Exception:
        raise HerdError("Invalid JSON body", status_code=400)

    model_name = body.get("model")
    if not model_name:
        raise HerdError("Missing 'model' field", status_code=400)

    model_name = resolve_pool_or_model(model_name)

    # Check if this model targets a registered cloud provider
    if ":" in model_name:
        from herd.core.config import settings

        settings.reload()
        parts = model_name.split(":", 1)
        if parts[0] in settings.providers:
            provider = parts[0]
            target_model = parts[1]
            return await proxy_to_cloud(
                provider, target_model, request, body, "/v1/chat/completions"
            )

    try:
        port = await manager.get_or_start_server(model_name, is_whisper=False, is_embedding=False)
    except FileNotFoundError as e:
        raise HerdError(str(e), status_code=404)

    return await proxy_to_port(port, "/v1/chat/completions", request, body_bytes, model_name)


@router.post("/v1/completions")
async def completions(request: Request):
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
    except Exception:
        raise HerdError("Invalid JSON body", status_code=400)

    model_name = body.get("model")
    if not model_name:
        raise HerdError("Missing 'model' field", status_code=400)

    model_name = resolve_pool_or_model(model_name)

    # Check if this model targets a registered cloud provider
    if ":" in model_name:
        from herd.core.config import settings

        settings.reload()
        parts = model_name.split(":", 1)
        if parts[0] in settings.providers:
            provider = parts[0]
            target_model = parts[1]
            return await proxy_to_cloud(provider, target_model, request, body, "/v1/completions")

    try:
        port = await manager.get_or_start_server(model_name, is_whisper=False, is_embedding=False)
    except FileNotFoundError as e:
        raise HerdError(str(e), status_code=404)

    return await proxy_to_port(port, "/v1/completions", request, body_bytes, model_name)


@router.post("/v1/embeddings")
async def embeddings(request: Request):
    body_bytes = await request.body()
    try:
        body = json.loads(body_bytes)
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
            provider = parts[0]
            target_model = parts[1]
            return await proxy_to_cloud(provider, target_model, request, body, "/v1/embeddings")

    try:
        port = await manager.get_or_start_server(model_name, is_whisper=False, is_embedding=True)
    except FileNotFoundError as e:
        raise HerdError(str(e), status_code=404)

    return await proxy_to_port(port, "/v1/embeddings", request, body_bytes, model_name)


@router.post("/v1/audio/transcriptions")
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
        raise HerdError(str(e), status_code=404)

    return await whisper_proxy(
        port,
        file,
        language,
        temperature,
        response_format,
        translate=False,
        model=model,
    )


@router.post("/v1/audio/translations")
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
        raise HerdError(str(e), status_code=404)

    return await whisper_proxy(
        port,
        file,
        language,
        temperature,
        response_format,
        translate=True,
        model=model,
    )
