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

You can also run an **interactive RAG chat** by passing the `--context` / `-c` option with an embedding model. Every message you type will be semantically matched against your indexed documents, and the relevant context will be automatically and transparently injected into the chat:
```bash
herd run unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M --context sentence-transformers/all-MiniLM-L6-v2:Q8_0
```

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
# Stop a specific running model
herd stop unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M

# Stop all running models at once
herd stop --all
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

# Start the gateway and expose it globally via Cloudflare Tunnel
herd serve --public
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

### 12. Smart Model Recommendations (`herd suggest`)
Audits your active hardware (CPU, RAM, GPU VRAM) and outputs custom-tailored LLM and Whisper model suggestions from Hugging Face that are guaranteed to run fast and fit comfortably on your machine:
```bash
herd suggest
```

### 13. Speech-to-Text Transcription (`herd transcribe`)
Transcribes or translates audio files locally into raw text (`.txt`) or standard subtitle files (`.srt` / `.vtt`) using Whisper:
```bash
# Transcribe to plain text using first auto-detected local Whisper model
herd transcribe meeting.wav

# Transcribe using specific model and save as WebVTT subtitles
herd transcribe lecture.wav --model ggerganov/whisper.cpp:ggml-base.en.bin --format vtt --output lecture.vtt
```

### 14. Local Network Exposer & Pairing (`herd share`)
Exposes connection details and prints a scanable terminal ASCII QR code to quickly pair and configure mobile/desktop clients (e.g. Chatbox, LibreChat) to your local gateway:
```bash
# Display local network API base URL and Dashboard URL
herd share

# Generate an ASCII QR code for easy mobile scan/pair
herd share --qr

# Expose to public internet using Cloudflare Tunnel
herd share --public
```

### 15. Inspect Model Metadata (`herd show`)
Displays detailed file locations, creation dates, sizes, and parsed GGUF header metadata (such as internal base architecture, baseline name, and quantization values) for any local model:
```bash
herd show Qwen/Qwen3.5-0.8B:Q8_0
```

### 16. Log Directory Maintenance (`herd clean`)
Cleans up inactive model logs in `~/.herd/logs/` while automatically preserving logs for currently active model servers and the API gateway itself:
```bash
# Audit and prompt for confirmation before deleting logs
herd clean

# Force clean logs immediately
herd clean --force
```

### 17. Document Indexing (`herd index`)
Recursively chunks and embeds files in a directory, storing their text and vector embeddings in the local SQLite vector database (`~/.herd/embeddings.db`):
```bash
herd index /path/to/my/documents -m sentence-transformers/all-MiniLM-L6-v2:Q8_0
```

### 18. Semantic QA over Indexed Docs (`herd ask`)
Performs a semantic query, retrieves the most relevant indexed document chunks, and streams the answer from the local LLM using the retrieved context:
```bash
herd ask Qwen/Qwen3.5-0.8B:Q8_0 "How do I configure the API Gateway?" -m sentence-transformers/all-MiniLM-L6-v2:Q8_0
```

### 19. Search Hugging Face Hub (`herd search`)
Searches the Hugging Face Hub for GGUF model repositories matching the query, sorted by download counts and formatted in a structured summary table:
```bash
herd search qwen --limit 5
```

### 20. Compress GGUF Models (`herd quantize`)
Compresses a GGUF model file locally using the precompiled `llama-quantize` compilation tool:
```bash
herd quantize model-fp16.gguf model-q4_k_m.gguf Q4_K_M
```

### 21. Real-Time Resource Monitor (`herd top`)
Opens a live terminal HUD showing running model processes, ports, CPU percentage bar meters, RSS memory footprints, and idle timers:
```bash
herd top
```

### 22. RAG Index Database Manager (`herd db`)
Provides utilities to inspect, query, and prune the local vector database index (`~/.herd/embeddings.db`):
```bash
# List all indexed files, directories, and chunk counts
herd db list

# Perform a raw semantic query and view matched source text chunks
herd db search "API config instructions" -m sentence-transformers/all-MiniLM-L6-v2:Q8_0

# Remove files under a specific directory from the database index
herd db remove /path/to/my/documents
```

### 23. Local Shell Copilot (`herd copilot`)
Translates natural language instructions into local shell commands, explains what they do, and runs them upon user confirmation:
```bash
herd copilot "find all python files recursively and count their lines of code"
```

### 24. Automated Local Git Commits (`herd commit`)
Inspects unstaged or staged modifications in the current Git repository using `git diff`, queries the local LLM to generate a clear Conventional Commit message, and commits changes upon confirmation:
```bash
herd commit
```

### 25. Configuration Defaults (`herd config`)
Herd allows configuring global default models, ports, and execution timeouts in `~/.herd/config.json`. These defaults are automatically used as fallback settings across all commands (`run`, `copilot`, `commit`, `index`, `ask`, `transcribe`) so you don't need to specify model flags every time:
```bash
# Display a formatted table of all current settings and defaults
herd config show

# Set the default LLM model for chatting, copilot, and commits
herd config set default_llm Qwen/Qwen3.5-0.8B:Q8_0

# Set the default embedding model for indexing and semantic context queries
herd config set default_embedding sentence-transformers/all-MiniLM-L6-v2:Q8_0

# Set the default Whisper model for STT transcribing
herd config set default_whisper ggerganov/whisper.cpp:ggml-base.en.bin

# Route all local CLI commands directly to a remote Herd gateway
herd config set remote_gateway http://192.168.1.100:11434

# Configure a remote cloud model provider (e.g. Groq, OpenAI, DeepSeek)
herd config set-provider groq --api-key gsk_yourAPIkeyHere

# Remove a cloud provider configuration
herd config remove-provider groq
```

Once configured, you can route chat, text completion, and embedding requests directly to remote cloud providers by prefixing the model identifier with `provider:`. If the prefix is not a configured provider, Herd automatically falls back to local GGUF execution:
```bash
# Route chat completions to Groq
herd run groq:llama-3.1-70b-versatile

# Route chat completions to OpenAI
herd run openai:gpt-4o-mini
```


### 26. Self-Healing Command Debugger (`herd heal`)
Runs a shell command in real-time, intercepts process crashes (non-zero exit codes), retrieves the terminal failure traceback, queries the local LLM for diagnosis, and proposes/executes the fix upon confirmation:
```bash
herd heal "python3 app.py"
```

### 27. Multimodal Vision Analyzer (`herd watch`)
Analyzes local image paths or remote image URLs using vision-language models (VLM) with automatic multimodal projector loading support:
```bash
# Analyze a local screenshot
herd watch screenshot.png "What error code is shown in the image?"

# Query a remote image URL
herd watch https://example.com/logo.png "Describe the design and colors."
```

### 28. Autonomous AI Agent (`herd agent`)
Launches an autonomous local AI agent loop (Thought -> Action -> Observation -> Repeat) that can read/write files, list directories, and execute shell commands locally to satisfy a high-level task:
```bash
herd agent "Find all python files in this project, scan for TODO comments, and write them into a TODO.md markdown table."
```

### 29. Local Reverse Proxy Gateway (`herd proxy`)
Spawns a local reverse-proxy gateway server (port `11434`) that transparently forwards all API endpoints (such as `/v1/chat/completions`, model load/unload commands, and RAG operations) to a remote Herd instance. It fully preserves server-sent event (SSE) streaming, allowing local client apps (like Chatbox or LibreChat) to offload model execution to a remote GPU host:
```bash
herd proxy http://192.168.1.100:11434
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

### Hugging Face Model Search (`GET /v1/hf/search`)
Query the Hugging Face Hub directly for GGUF model files matching a keyword search query:
```bash
curl "http://127.0.0.1:11434/v1/hf/search?query=Qwen2.5-0.5B"
```

### Background Model Download (`POST /v1/models/pull`)
Initiate a model download in the background. Tracks current download progress dynamically:
```bash
# Pull model files
curl -X POST http://127.0.0.1:11434/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "unsloth/Qwen3.5-0.8B-GGUF"}'

# Query download progress
curl http://127.0.0.1:11434/v1/models/pull/status
```

### RAG Vector Indexing & Search (`POST /v1/db/index` & `POST /v1/db/search`)
Index a local folder recursively into the vector database, or query it semantically using the local embedding model:
```bash
# Index directory chunks
curl -X POST http://127.0.0.1:11434/v1/db/index \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "/path/to/docs",
    "model": "sentence-transformers/all-MiniLM-L6-v2:Q8_0"
  }'

# Semantic RAG search
curl -X POST http://127.0.0.1:11434/v1/db/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do we initialize the database?",
    "model": "sentence-transformers/all-MiniLM-L6-v2:Q8_0",
    "limit": 3
  }'
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
