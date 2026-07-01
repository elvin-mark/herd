<p align="center">
  <img src="assets/logo.jpg" alt="Herd Logo" width="280" />
</p>

# Herd 🦙

Herd is a local AI model coordinator and API gateway inspired by Ollama. The name comes from the idea of running a **"herd" of llamas** (Llama LLM processes) concurrently on your system. It enables you to download GGUF (LLM) and GGML/GGUF (Whisper) models directly from Hugging Face, run multiple models concurrently in separate background processes (powered by `llama.cpp` and `whisper.cpp`), and access them all through a single OpenAI-compatible API gateway.

## Features

- **Hugging Face Hub Integration**: Download models directly from HF using `author/repo` and optional tags (e.g. `unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M`).
- **Unified API Gateway**: Routes incoming requests to the correct model process based on the `model` request field.
- **On-Demand Loading & Lifecycle Management**: Automatically starts model processes when requests arrive, and stops them after a period of inactivity (default 5 minutes) to conserve RAM/VRAM.
- **Embedding Support**: Start LLMs with embedding capabilities via the `--embedding` flag.
- **Whisper Integration**: Supports audio transcription and translation endpoints using `whisper-server`.
- **Interactive Chat CLI**: Drop into a real-time streaming chat session with any downloaded model directly in your terminal.

---

## Installation

1. **Prerequisites**:
   - Python 3.10 or higher
   - `llama-server` (from [llama.cpp](https://github.com/ggerganov/llama.cpp)) and `whisper-server` (from [whisper.cpp](https://github.com/ggerganov/whisper.cpp)) must be installed on your system path, or their paths set via environment variables.

2. **Clone & Install**:
   ```bash
   # Clone the repository
   git clone <repo-url>
   cd herd

    # Create virtual environment and install (Standard Python)
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

    # OR install using uv (Blazing-Fast)
    uv venv
    source .venv/bin/activate
    uv pip install -e .
   ```

---

## CLI Usage

### 1. Download a Model (`herd pull`)
Download models from Hugging Face. If no tag is specified, Herd will list the available files and prompt you to choose one:
```bash
herd pull unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
```

### 2. Run a Model and Chat (`herd run`)
Starts the gateway in the background (if not already running), loads the model, and enters an interactive chat session:
```bash
herd run unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
```
If the model has not been downloaded yet, Herd will ask if you want to pull it.

You can specify a custom idle timeout (in seconds) before the model process is automatically stopped to free up RAM/VRAM. Pass `0` to keep the model running indefinitely:
```bash
# Set idle timeout to 10 minutes (600 seconds)
herd run unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M --idle-timeout 600

# Prevent the model from ever auto-stopping
herd run unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M --idle-timeout 0
```

To load a Whisper model as a transcription server, use the `--whisper` flag:
```bash
herd run ggerganov/whisper.cpp:ggml-base.en.bin --whisper
```

### 3. List Downloaded Models (`herd list`)
View all local models stored under `HERD_HOME`, organized by provider. You can optionally filter them by name or provider:
```bash
# List all downloaded models
herd list

# Filter models by name (case-insensitive substring)
herd list qwen

# Filter models by provider (e.g. huggingface)
herd list --provider huggingface
```

### 4. View Active Model processes (`herd ps`)
See which models are currently loaded, their internal ports, CPU usage, RAM utilization, and idle time:
```bash
herd ps
```

### 5. View Performance Statistics (`herd stats`)
See cumulative request counts, error counts, prompt tokens processed, generation tokens generated, average latency, and average generation speed (tokens per second) across all models:
```bash
herd stats
```

### 6. Stop a Running Model (`herd stop`)
Manually unload a model and terminate its process:
```bash
herd stop unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M
```

### 7. View Server and Model Logs (`herd logs`)
View or live-tail the runtime logs of the gateway or any active model process. Press `Ctrl+C` to exit:
```bash
# View the last 20 lines of the main API gateway logs
herd logs

# Live-follow the last 50 lines of the gateway logs in real-time
herd logs -f -n 50

# Live-follow the logs for a specific running model process
herd logs unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M -f
```

### 8. Start the API Gateway (`herd serve`)
Manually start the gateway in the foreground. Use `--host 0.0.0.0` to expose the gateway and its models to other machines on your local network:
```bash
# Bind to localhost (default)
herd serve --port 11434

# Bind to all network interfaces for local network access
herd serve --host 0.0.0.0 --port 11434
```

### 9. Build and Configure Binaries Locally (`herd setup`)
If you do not have `llama-server` or `whisper-server` installed on your system PATH, Herd can automatically clone, compile, and configure the binaries locally in your `HERD_HOME` directory. Requires `git` and `cmake` to be installed on your system:
```bash
# Clone and build llama-server and whisper-server locally with CPU support
herd setup

# Clone and compile with CUDA/GPU support (NVIDIA GPUs)
herd setup --cuda

# Specify a custom directory to clone and build the repositories
herd setup --dir ./my-builds
```
This command automatically writes the compiled absolute binary paths to a configuration file `~/.herd/config.json`, which Herd loads at startup.

### 10. Run System Diagnostics (`herd doctor`)
Audits your system environment, hardware compatibility, compiling prerequisites, and gateway server status. It checks CPU AVX2/AVX512/Neon instruction capabilities and NVIDIA GPU VRAM availability:
```bash
herd doctor
```

### 11. Run Model Benchmarks (`herd benchmark`)
Benchmarks an LLM's prompt ingestion latency (TTFT), generation speed, and system memory/CPU footprint using standardized prompts (or custom prompts via `-p`):
```bash
# Run benchmark suite (3 rounds per prompt by default)
herd benchmark unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M

# Run with custom prompts and 2 rounds
herd benchmark unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M -p "Write a song.,Tell a joke." -r 2
```

### Interactive Chat Slash Commands
During an active CLI chat session (launched via `herd run <model>`), you can use the following interactive slash commands:
*   `/help` — Displays the help menu listing all available chat commands.
*   `/clear` or `/reset` — Clears the conversation history (context window).
*   `/system <prompt>` — Dynamically sets or updates the system prompt.
*   `/export [filename]` — Exports the current chat transcript to a markdown file (default: `chat_export.md`).
*   `/exit` or `/quit` — Exits the chat session.

---

## API Usage & Control Center

The Herd gateway listens on port `11434` (by default) and exposes a web control panel as well as standard OpenAI-compatible endpoints:

### Web Control Center Dashboard
You can monitor active models, CPU/RAM utilization, view cumulative statistics, and manage your local model library (with load and unload buttons) via a beautiful, real-time GUI in your browser:
* **URL**: `http://localhost:11434` or `http://localhost:11434/dashboard`

### OpenAI-Compatible API Endpoints

### Chat Completions (`POST /v1/chat/completions`)
```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M",
    "messages": [{"role": "user", "content": "Explain gravity in one sentence."}],
    "stream": true
  }'
```

### Audio Transcriptions (`POST /v1/audio/transcriptions`)
```bash
curl http://127.0.0.1:11434/v1/audio/transcriptions \
  -F "file=@/path/to/audio.wav" \
  -F "model=ggerganov/whisper.cpp:ggml-base.en.bin" \
  -F "language=en"
```

### Prometheus Scraping Metrics (`GET /metrics`)
Exposes standard Prometheus plain-text metrics (e.g. CPU, memory, requests total, token counts, and request durations) for third-party dashboards (Grafana, etc.):
```bash
curl http://127.0.0.1:11434/metrics
```

---

## Configuration

You can customize Herd's behavior using the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HERD_HOME` | Directory where models and logs are stored | `~/.herd` |
| `HERD_HOST` | Host IP address to bind the Herd API gateway | `127.0.0.1` |
| `HERD_PORT` | Port for the central Herd API gateway | `11434` |
| `HERD_IDLE_TIMEOUT` | Seconds of inactivity before stopping a model process | `300` (5 minutes) |
| `LLAMA_SERVER_BIN` | Custom path to the `llama-server` binary | Discovered in `PATH` |
| `WHISPER_SERVER_BIN` | Custom path to the `whisper-server` binary | Discovered in `PATH` |
