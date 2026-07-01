# Herd API Reference

Herd exposes a central API Gateway on port `11434` (by default) that acts as a proxy for local model processes. It supports standard OpenAI-compatible endpoints as well as lifecycle control, observability, and diagnostics endpoints.

---

## 🦙 OpenAI-Compatible Endpoints

### 1. Chat Completions (`POST /v1/chat/completions`)
Creates a model response for a given chat conversation. If the requested model is not running, the gateway will load it on demand.

*   **Content-Type**: `application/json`
*   **Request Body**:
    *   `model` (string, required): The Hugging Face identifier or file path of the local GGUF model (e.g. `unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M`).
    *   `messages` (array, required): List of message objects. Each object should have `role` (`user`, `assistant`, `system`) and `content` (string).
    *   `stream` (boolean, optional): If `true`, returns a stream of Server-Sent Events (SSE) chunks ending with `data: [DONE]`.
    *   *Other standard OpenAI parameters (temperature, max_tokens, etc.) are forwarded directly to the backend `llama-server`.*

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "stream": false
  }'
```

#### Example Response (Non-streaming):
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "Hello! How can I help you today?",
        "role": "assistant"
      }
    }
  ],
  "created": 1719878400,
  "id": "chatcmpl-...",
  "model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 9,
    "prompt_tokens": 18,
    "total_tokens": 27
  }
}
```

---

### 2. Legacy Completions (`POST /v1/completions`)
Generates completions for prompt strings.

*   **Content-Type**: `application/json`
*   **Request Body**:
    *   `model` (string, required): GGUF model identifier.
    *   `prompt` (string, required): The prompt to complete.
    *   `stream` (boolean, optional): Stream tokens via SSE.

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    "prompt": "The capital of France is",
    "stream": false
  }'
```

---

### 3. Embeddings (`POST /v1/embeddings`)
Generates vector representations of input text.
> **Note**: Herd automatically restarts the backend `llama-server` with the `--embedding` flag if it was running without it.

*   **Content-Type**: `application/json`
*   **Request Body**:
    *   `model` (string, required): Embedding model identifier.
    *   `input` (string or array of strings, required): The input text to embed.

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2:Q8_0",
    "input": "Vector search is powerful."
  }'
```

---

### 4. Audio Transcriptions (`POST /v1/audio/transcriptions`)
Transcribes speech audio into text using Whisper.

*   **Content-Type**: `multipart/form-data`
*   **Request Parameters**:
    *   `file` (file, required): The binary audio file (typically `.wav` format, 16kHz mono).
    *   `model` (string, required): The Whisper model identifier (e.g. `ggerganov/whisper.cpp:ggml-base.en.bin`).
    *   `language` (string, optional): Target audio language code (e.g. `en`, `es`). Defaults to `auto` (auto-detection).
    *   `temperature` (float, optional): Temperature controls for output diversity.
    *   `response_format` (string, optional): Request format. Supports `json` (OpenAI format) or `text` (raw plain text).

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/audio/transcriptions \
  -F "file=@/path/to/sample.wav" \
  -F "model=ggerganov/whisper.cpp:ggml-base.en.bin" \
  -F "language=en"
```

---

### 5. Audio Translations (`POST /v1/audio/translations`)
Translates non-English speech audio into English text.

*   **Content-Type**: `multipart/form-data`
*   **Request Parameters**: Same as `/v1/audio/transcriptions` but automatically forces translation to English.

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/audio/translations \
  -F "file=@/path/to/spanish_audio.wav" \
  -F "model=ggerganov/whisper.cpp:ggml-base.bin"
```

---

## 🛠️ Model Management Endpoints

### 6. List Downloaded Models (`GET /v1/models`)
Lists all models currently downloaded on disk that are ready to run.

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/models
```

#### Example Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "unsloth/Qwen3.5-0.8B-GGUF",
      "object": "model",
      "created": 1719878500,
      "owned_by": "huggingface"
    }
  ]
}
```

---

### 7. Active Models (`GET /v1/models/active`)
Lists currently running model server processes and their system resource footprint.

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/models/active
```

#### Example Response:
```json
[
  {
    "model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    "port": 42051,
    "is_whisper": false,
    "is_embedding": false,
    "last_accessed": 1719878600.123,
    "idle_seconds": 45,
    "log_path": "/home/user/.herd/logs/unsloth_Qwen3.5-0.8B-GGUF_Q4_K_M.log",
    "cpu_percent": 12.5,
    "memory_bytes": 1610612736,
    "memory_str": "1.50 GB"
  }
]
```

---

### 8. Explicitly Load Model (`POST /v1/models/load`)
Explicitly spawns a server process for a model. If the model is already running, it updates the settings and keeps the process alive.

*   **Content-Type**: `application/json`
*   **Request Body**:
    *   `model` (string, required): The model identifier.
    *   `is_whisper` (boolean, optional): Launch as a Whisper server. Defaults to `false`.
    *   `is_embedding` (boolean, optional): Enable embeddings flags. Defaults to `false`.
    *   `idle_timeout` (integer, optional): Override the idle timeout in seconds (use `0` to run indefinitely).

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    "idle_timeout": 600
  }'
```

---

### 9. Explicitly Unload Model (`POST /v1/models/unload`)
Stops the process of a running model.

*   **Content-Type**: `application/json`
*   **Request Body**:
    *   `model` (string, required): The model identifier.

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/models/unload \
  -H "Content-Type: application/json" \
  -d '{"model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M"}'
```

---

### 10. Cumulative Model Stats (`GET /v1/models/stats`)
Retrieves cumulative tracking data for requests, throughput, speed, and latencies.

#### Example Request:
```bash
curl http://127.0.0.1:11434/v1/models/stats
```

#### Example Response:
```json
{
  "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M": {
    "requests": 14,
    "errors": 0,
    "prompt_tokens": 284,
    "completion_tokens": 1253,
    "total_tokens": 1537,
    "avg_latency_sec": 1.45,
    "avg_speed_tok_sec": 24.5,
    "endpoints": {
      "/v1/chat/completions": 14
    }
  }
}
```

---

## 📈 Diagnostics & Observability

### 11. Prometheus Scraper (`GET /metrics`)
Exposes performance, CPU, memory, request, and token metrics in plain text suitable for scraping by a Prometheus collector.

#### Example Request:
```bash
curl http://127.0.0.1:11434/metrics
```

#### Example Response:
```text
# HELP herd_active_models Number of active running models.
# TYPE herd_active_models gauge
herd_active_models 1
# HELP herd_model_cpu_percent CPU usage percentage of the active model process.
# TYPE herd_model_cpu_percent gauge
herd_model_cpu_percent{model="unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",port="42051"} 12.5
# HELP herd_model_memory_bytes RAM usage in bytes of the active model process.
# TYPE herd_model_memory_bytes gauge
herd_model_memory_bytes{model="unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",port="42051"} 1610612736
# HELP herd_requests_total Total number of API requests sent to Herd.
# TYPE herd_requests_total counter
herd_requests_total{model="unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",endpoint="/v1/chat/completions"} 14
```

---

### 12. Server Health check (`GET /health`)
Returns the gateway status. Used by CLI to verify if the gateway is running.

#### Example Request:
```bash
curl http://127.0.0.1:11434/health
```

#### Example Response:
```json
{
  "status": "ok"
}
```
