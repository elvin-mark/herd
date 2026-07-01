import os
import sys
import time
import json
import asyncio
import subprocess
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
import httpx
import uvicorn

import shutil
from herd.core.config import (
    HERD_HOST,
    HERD_PORT,
    HERD_LOGS_DIR,
    HERD_MODELS_DIR,
    HERD_HOME,
    LLAMA_SERVER_BIN,
    WHISPER_SERVER_BIN,
    IDLE_TIMEOUT,
)
from herd.services.downloader import (
    list_hf_repository_files,
    download_file,
    parse_model_identifier,
    resolve_model_path,
)

app = typer.Typer(
    name="herd",
    help="Herd - A CLI and API gateway for running local LLM and speech-to-text models (similar to Ollama).",
    no_args_is_help=True,
)
console = Console()


def is_gateway_running() -> bool:
    """Checks if the Herd gateway server is currently running."""
    host = HERD_HOST
    if host == "0.0.0.0":
        host = "127.0.0.1"
    try:
        response = httpx.get(f"http://{host}:{HERD_PORT}/health", timeout=1.0)
        return response.status_code == 200
    except Exception:
        return False


def auto_start_gateway() -> bool:
    """Starts the Herd gateway server in the background if it isn't running."""
    if is_gateway_running():
        return True

    console.print(
        "[yellow]Herd API gateway is not running. Starting gateway in background...[/yellow]"
    )
    log_path = os.path.join(HERD_LOGS_DIR, "gateway.log")
    os.makedirs(HERD_LOGS_DIR, exist_ok=True)

    # Open log file in append mode
    log_file = open(log_path, "a")

    try:
        # Launch uvicorn gateway in background, detached
        subprocess.Popen(
            [sys.executable, "-m", "herd.cli", "serve"],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
    except Exception as e:
        console.print(f"[red]Failed to spawn Herd gateway subprocess: {e}[/red]")
        log_file.close()
        return False

    log_file.close()

    # Poll gateway /health endpoint to wait for startup (up to 10 seconds)
    for _ in range(20):
        time.sleep(0.5)
        if is_gateway_running():
            console.print(
                "[green]Herd API gateway started successfully in the background.[/green]"
            )
            return True

    console.print(
        f"[red]Herd gateway failed to start within timeout. Check logs at: {log_path}[/red]"
    )
    return False


def get_local_models_info():
    """Scans local HERD_MODELS_DIR directly to find downloaded GGUF/bin files, organized by provider."""
    models = []
    if not os.path.exists(HERD_MODELS_DIR):
        return models

    for provider in os.listdir(HERD_MODELS_DIR):
        provider_path = os.path.join(HERD_MODELS_DIR, provider)
        if not os.path.isdir(provider_path):
            continue

        if provider == "huggingface":
            for author in os.listdir(provider_path):
                author_path = os.path.join(provider_path, author)
                if not os.path.isdir(author_path):
                    continue
                for repo in os.listdir(author_path):
                    repo_path = os.path.join(author_path, repo)
                    if not os.path.isdir(repo_path):
                        continue

                    for root, _, files in os.walk(repo_path):
                        for file in files:
                            if file.endswith(".gguf") or file.endswith(".bin"):
                                full_path = os.path.join(root, file)
                                size_bytes = os.path.getsize(full_path)
                                size_gb = size_bytes / (1024 * 1024 * 1024)
                                size_str = (
                                    f"{size_gb:.2f} GB"
                                    if size_gb >= 1.0
                                    else f"{size_bytes / (1024 * 1024):.2f} MB"
                                )

                                import re

                                name_no_ext, _ = os.path.splitext(file)
                                match = re.search(
                                    r"([qI]?[0-9]_[A-Za-z0-9_]+)",
                                    name_no_ext,
                                    re.IGNORECASE,
                                )
                                tag = match.group(1) if match else name_no_ext

                                models.append(
                                    {
                                        "provider": provider,
                                        "name": f"{author}/{repo}:{tag}",
                                        "filename": file,
                                        "size": size_str,
                                        "path": full_path,
                                    }
                                )
        else:
            for root, _, files in os.walk(provider_path):
                for file in files:
                    if file.endswith(".gguf") or file.endswith(".bin"):
                        full_path = os.path.join(root, file)
                        size_bytes = os.path.getsize(full_path)
                        size_gb = size_bytes / (1024 * 1024 * 1024)
                        size_str = (
                            f"{size_gb:.2f} GB"
                            if size_gb >= 1.0
                            else f"{size_bytes / (1024 * 1024):.2f} MB"
                        )

                        models.append(
                            {
                                "provider": provider,
                                "name": file,
                                "filename": file,
                                "size": size_str,
                                "path": full_path,
                            }
                        )
    return models


async def pull_model_async(model_name: str):
    """Pulls a model GGUF/bin file from Hugging Face."""
    try:
        author, repo, tag = parse_model_identifier(model_name)
    except ValueError as e:
        console.print(f"[red]Error parsing model: {e}[/red]")
        raise typer.Exit(1)

    console.print(
        f"Fetching repository details for [bold cyan]{author}/{repo}[/bold cyan] from Hugging Face..."
    )
    try:
        files = await list_hf_repository_files(author, repo)
    except Exception as e:
        console.print(f"[red]Failed to list Hugging Face repository files: {e}[/red]")
        raise typer.Exit(1)

    # Filter files ending in .gguf or .bin
    model_files = [f for f in files if f.endswith(".gguf") or f.endswith(".bin")]
    if not model_files:
        console.print(
            f"[red]No GGUF or BIN model files found in repository {author}/{repo}.[/red]"
        )
        raise typer.Exit(1)

    chosen_file = None
    if tag:
        # Search for file matching the tag
        matches = [f for f in model_files if tag.lower() in f.lower()]
        if not matches:
            console.print(
                f"[red]No files matching tag '{tag}' found. Available model files:[/red]"
            )
            for f in model_files:
                console.print(f" - {f}")
            raise typer.Exit(1)
        chosen_file = matches[0]
    else:
        # Interactive mode: let the user choose
        console.print(
            "\nMultiple model files available. Please select one to download:"
        )
        for idx, f in enumerate(model_files):
            console.print(f"[[bold green]{idx}[/bold green]] {f}")

        choice = typer.prompt("\nEnter model index", type=int)
        if choice < 0 or choice >= len(model_files):
            console.print("[red]Invalid index selected.[/red]")
            raise typer.Exit(1)
        chosen_file = model_files[choice]

    # Build download URL and destination path
    download_url = f"https://huggingface.co/{author}/{repo}/resolve/main/{chosen_file}"
    dest_path = os.path.join(HERD_MODELS_DIR, "huggingface", author, repo, chosen_file)

    console.print(f"Downloading file: [bold cyan]{chosen_file}[/bold cyan]")
    try:
        await download_file(download_url, dest_path, chosen_file)
        console.print(
            f"[green]Successfully downloaded and saved model to: {dest_path}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Failed to download file: {e}[/red]")
        raise typer.Exit(1)


async def stream_chat_completions(model_name: str, messages: list) -> str:
    """Sends a streaming chat completion request to Herd gateway."""
    url = f"http://127.0.0.1:{HERD_PORT}/v1/chat/completions"
    payload = {"model": model_name, "messages": messages, "stream": True}

    assistant_response = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                err_body = await response.aread()
                try:
                    err_json = json.loads(err_body)
                    detail = err_json.get("error", "Unknown error")
                except Exception:
                    detail = err_body.decode()
                raise RuntimeError(
                    f"Gateway returned error {response.status_code}: {detail}"
                )

            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                assistant_response += content
                    except Exception:
                        pass
    print()
    return assistant_response


async def chat_interactive(model_name: str):
    """Launches the CLI chat session loops."""
    console.print(
        f"\n[bold green]Chatting with {model_name} (Herd Gateway)[/bold green]"
    )
    console.print("Type /exit or /quit to end. Press Ctrl+C to stop generation.\n")

    messages = []
    while True:
        try:
            user_input = typer.prompt(">>> ", prompt_suffix="").strip()
            if not user_input:
                continue
            if user_input.lower() in ["/exit", "/quit"]:
                break

            messages.append({"role": "user", "content": user_input})
            print("Response: ", end="")

            try:
                assistant_response = await stream_chat_completions(model_name, messages)
                messages.append({"role": "assistant", "content": assistant_response})
            except KeyboardInterrupt:
                print("\n[yellow]Generation interrupted.[/yellow]")
                messages.append(
                    {"role": "assistant", "content": "[Generation Interrupted]"}
                )
            except Exception as e:
                console.print(f"\n[red]Error during generation: {e}[/red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting chat session...[/yellow]")
            break


@app.command()
def pull(
    model_name: str = typer.Argument(
        ...,
        help="Model identifier format 'author/repo[:tag]' (e.g. unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M)",
    ),
):
    """Downloads a GGUF or BIN model from Hugging Face."""
    asyncio.run(pull_model_async(model_name))


@app.command()
def run(
    model_name: str = typer.Argument(..., help="Model identifier to run."),
    whisper: bool = typer.Option(
        False,
        "--whisper",
        "-w",
        help="Run as a speech-to-text whisper model server.",
    ),
    embedding: bool = typer.Option(
        False,
        "--embedding",
        "-e",
        help="Enable embeddings flag for llama-server.",
    ),
    idle_timeout: Optional[int] = typer.Option(
        None,
        "--idle-timeout",
        "-t",
        help="Idle timeout in seconds before stopping the model process (0 to keep running indefinitely).",
    ),
):
    """Starts a model and opens an interactive chat (or runs transcription server)."""
    # 1. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # 2. Check if model exists locally. If not, prompt to download
    try:
        resolve_model_path(model_name)
    except FileNotFoundError:
        console.print(f"[yellow]Model '{model_name}' not found locally.[/yellow]")
        confirm = typer.confirm("Would you like to pull/download it now?")
        if confirm:
            asyncio.run(pull_model_async(model_name))
        else:
            console.print("[red]Aborted.[/red]")
            raise typer.Exit(1)

    # 3. Load model in gateway
    console.print(f"Loading [bold cyan]{model_name}[/bold cyan] in Herd gateway...")
    url = f"http://127.0.0.1:{HERD_PORT}/v1/models/load"
    try:
        response = httpx.post(
            url,
            json={
                "model": model_name,
                "is_whisper": whisper,
                "is_embedding": embedding,
                "idle_timeout": idle_timeout,
            },
            timeout=45.0,
        )
        if response.status_code != 200:
            console.print(f"[red]Failed to load model: {response.text}[/red]")
            raise typer.Exit(1)

        data = response.json()
        port = data["port"]
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1)

    # 4. Enter chat REPL or display server status
    if whisper or "whisper" in model_name.lower():
        console.print("\n[bold green]Whisper model loaded successfully![/bold green]")
        console.print(
            f"Whisper server running internally on port [bold cyan]{port}[/bold cyan]."
        )
        console.print("You can send transcription requests to the Gateway:")
        console.print(
            f"  [bold white]POST http://127.0.0.1:{HERD_PORT}/v1/audio/transcriptions[/bold white]"
        )
        console.print(
            f"  [bold white]POST http://127.0.0.1:{HERD_PORT}/v1/audio/translations[/bold white]"
        )
    elif embedding or "embedding" in model_name.lower() or "bert" in model_name.lower():
        console.print("\n[bold green]Embedding model loaded successfully![/bold green]")
        console.print(
            f"Model server running internally on port [bold cyan]{port}[/bold cyan]."
        )
        console.print("You can send embedding requests to the Gateway:")
        console.print(
            f"  [bold white]POST http://127.0.0.1:{HERD_PORT}/v1/embeddings[/bold white]"
        )
    else:
        asyncio.run(chat_interactive(model_name))


@app.command(name="list")
def list_models(
    filter_query: Optional[str] = typer.Argument(
        None, help="Filter models by name (case-insensitive substring search)."
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="Filter models by provider (e.g. huggingface).",
    ),
):
    """Lists all downloaded models present on disk, optionally filtered."""
    models = get_local_models_info()

    if provider:
        models = [
            m for m in models if m.get("provider", "").lower() == provider.lower()
        ]

    if filter_query:
        models = [
            m
            for m in models
            if filter_query.lower() in m["name"].lower()
            or filter_query.lower() in m["filename"].lower()
        ]

    if not models:
        if filter_query or provider:
            console.print(
                "[yellow]No models matched the specified filter criteria.[/yellow]"
            )
        else:
            console.print(
                "[yellow]No models found. Use 'herd pull <model_name>' to download some.[/yellow]"
            )
        return

    table = Table(title="Downloaded Models")
    table.add_column("Provider", style="blue")
    table.add_column("Model Name", style="cyan", no_wrap=True)
    table.add_column("Filename", style="magenta")
    table.add_column("Size", style="green")

    for m in models:
        table.add_row(m.get("provider", "local"), m["name"], m["filename"], m["size"])

    console.print(table)


@app.command()
def ps():
    """Lists currently active running model processes in the gateway."""
    if not is_gateway_running():
        console.print("[yellow]Herd API gateway is not running.[/yellow]")
        return

    url = f"http://127.0.0.1:{HERD_PORT}/v1/models/active"
    try:
        response = httpx.get(url)
        response.raise_for_status()
        active = response.json()
    except Exception as e:
        console.print(f"[red]Failed to query active models: {e}[/red]")
        return

    if not active:
        console.print("[yellow]No models are currently running.[/yellow]")
        return

    table = Table(title="Active Running Models")
    table.add_column("Model", style="cyan")
    table.add_column("Port", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("CPU %", style="yellow")
    table.add_column("Memory", style="green")
    table.add_column("Idle Time", style="blue")

    for a in active:
        m_type = (
            "Whisper"
            if a["is_whisper"]
            else ("Embedding" if a["is_embedding"] else "LLM")
        )
        idle_str = f"{a['idle_seconds']}s"
        cpu_str = f"{a.get('cpu_percent', 0.0)}%"
        mem_str = a.get("memory_str", "0 MB")
        table.add_row(a["model"], str(a["port"]), m_type, cpu_str, mem_str, idle_str)

    console.print(table)


@app.command(name="stats")
def show_stats():
    """Displays cumulative request, token, and performance stats for all models."""
    if not is_gateway_running():
        console.print("[yellow]Herd API gateway is not running.[/yellow]")
        return

    url = f"http://127.0.0.1:{HERD_PORT}/v1/models/stats"
    try:
        response = httpx.get(url)
        response.raise_for_status()
        stats = response.json()
    except Exception as e:
        console.print(f"[red]Failed to query model stats: {e}[/red]")
        return

    if not stats:
        console.print(
            "[yellow]No stats collected yet. Send some requests first![/yellow]"
        )
        return

    table = Table(title="Herd Model Usage Statistics")
    table.add_column("Model", style="cyan")
    table.add_column("Requests (Err)", style="magenta")
    table.add_column("Prompt Tok", style="green")
    table.add_column("Gen Tok", style="green")
    table.add_column("Avg Latency", style="yellow")
    table.add_column("Avg Speed", style="bold green")

    for model, data in stats.items():
        req_str = f"{data['requests']} ({data['errors']})"
        lat_str = f"{data['avg_latency_sec']:.2f}s"
        speed_str = (
            f"{data['avg_speed_tok_sec']:.1f} tok/s"
            if data["avg_speed_tok_sec"] > 0
            else "N/A"
        )

        table.add_row(
            model,
            req_str,
            f"{data['prompt_tokens']:,}",
            f"{data['completion_tokens']:,}",
            lat_str,
            speed_str,
        )

    console.print(table)


@app.command()
def stop(model_name: str = typer.Argument(..., help="Model identifier to stop.")):
    """Stops a running model process."""
    if not is_gateway_running():
        console.print("[yellow]Herd API gateway is not running.[/yellow]")
        return

    url = f"http://127.0.0.1:{HERD_PORT}/v1/models/unload"
    try:
        response = httpx.post(url, json={"model": model_name})
        if response.status_code == 200:
            console.print(f"[green]Successfully stopped model '{model_name}'.[/green]")
        else:
            console.print(f"[red]Failed to stop model: {response.text}[/red]")
    except Exception as e:
        console.print(f"[red]Error stopping model: {e}[/red]")


@app.command(name="serve")
def serve(
    host: str = typer.Option(
        HERD_HOST,
        "--host",
        "-h",
        help="Host IP address to bind the gateway server to (use '0.0.0.0' for local network access).",
    ),
    port: int = typer.Option(
        HERD_PORT, "--port", "-p", help="Port to run the gateway server on."
    ),
):
    """Starts the central Herd API Gateway server."""
    # Ensure gateway port and host are set in env so other processes know about it
    os.environ["HERD_PORT"] = str(port)
    os.environ["HERD_HOST"] = host
    console.print(
        f"[bold green]Starting Herd API Gateway on {host}:{port}...[/bold green]"
    )
    # Correct path to the FastAPI app module under the new package layout
    uvicorn.run("herd.api.server:app", host=host, port=port, log_level="info")


@app.command()
def logs(
    model_name: Optional[str] = typer.Argument(
        None,
        help="Model identifier to view logs for. If omitted, tails the gateway logs.",
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow log output in real-time."
    ),
    lines: int = typer.Option(
        20, "--lines", "-n", help="Number of lines to show from the end of the logs."
    ),
):
    """Views or live-tails logs for a model process or the central gateway."""
    if model_name:
        model_safe = model_name.replace("/", "_").replace(":", "_")
        log_path = os.path.join(HERD_LOGS_DIR, f"{model_safe}.log")
        target_desc = f"Model '{model_name}'"
    else:
        log_path = os.path.join(HERD_LOGS_DIR, "gateway.log")
        target_desc = "Herd Gateway"

    if not os.path.exists(log_path):
        console.print(f"[red]No logs found at: {log_path}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Tailing last {lines} lines of {target_desc} logs...[/bold green]"
    )

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            from collections import deque

            last_lines = deque(f, maxlen=lines)
            for line in last_lines:
                print(line, end="")

            if follow:
                f.seek(0, 2)
                console.print(
                    "\n[bold yellow]--- Following logs (Press Ctrl+C to exit) ---[/bold yellow]\n"
                )
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    print(line, end="", flush=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Log tailing stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error reading logs: {e}[/red]")


@app.command()
def setup(
    dir_path: Optional[str] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory where llama.cpp and whisper.cpp will be cloned and compiled. Defaults to HERD_HOME/src.",
    ),
    cuda: bool = typer.Option(
        False, "--cuda", help="Compile llama.cpp and whisper.cpp with CUDA support."
    ),
):
    """Clones, compiles, and configures llama.cpp and whisper.cpp locally."""
    if not dir_path:
        dir_path = os.path.join(HERD_HOME, "src")

    os.makedirs(dir_path, exist_ok=True)

    git_bin = shutil.which("git")
    cmake_bin = shutil.which("cmake")
    if not git_bin:
        console.print("[red]Error: 'git' is not installed or not in PATH. Please install git first.[/red]")
        raise typer.Exit(1)
    if not cmake_bin:
        console.print("[red]Error: 'cmake' is not installed or not in PATH. Please install cmake first.[/red]")
        raise typer.Exit(1)

    llama_dir = os.path.join(dir_path, "llama.cpp")
    whisper_dir = os.path.join(dir_path, "whisper.cpp")

    # 1. Setup llama.cpp
    if not os.path.exists(llama_dir):
        console.print("[bold cyan]Cloning llama.cpp...[/bold cyan]")
        subprocess.run(
            [git_bin, "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", llama_dir],
            check=True
        )
    else:
        console.print("[yellow]llama.cpp directory already exists. Skipping clone.[/yellow]")

    console.print("[bold cyan]Compiling llama-server...[/bold cyan]")
    cmake_args = [cmake_bin, "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
    if cuda:
        cmake_args.append("-DGGML_CUDA=ON")

    cores = os.cpu_count() or 1
    subprocess.run(cmake_args, cwd=llama_dir, check=True)
    subprocess.run(
        [cmake_bin, "--build", "build", "--config", "Release", "--target", "llama-server", "--parallel", str(cores)],
        cwd=llama_dir,
        check=True
    )

    # 2. Setup whisper.cpp
    if not os.path.exists(whisper_dir):
        console.print("[bold cyan]Cloning whisper.cpp...[/bold cyan]")
        subprocess.run(
            [git_bin, "clone", "--depth", "1", "https://github.com/ggerganov/whisper.cpp.git", whisper_dir],
            check=True
        )
    else:
        console.print("[yellow]whisper.cpp directory already exists. Skipping clone.[/yellow]")

    console.print("[bold cyan]Compiling whisper-server...[/bold cyan]")
    whisper_cmake_args = [cmake_bin, "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
    if cuda:
        whisper_cmake_args.append("-DGGML_CUDA=ON")

    subprocess.run(whisper_cmake_args, cwd=whisper_dir, check=True)
    subprocess.run(
        [cmake_bin, "--build", "build", "--config", "Release", "--target", "whisper-server", "--parallel", str(cores)],
        cwd=whisper_dir,
        check=True
    )

    # 3. Configure binary paths
    llama_bin_path = os.path.abspath(os.path.join(llama_dir, "build", "bin", "llama-server"))
    whisper_bin_path = os.path.abspath(os.path.join(whisper_dir, "build", "bin", "whisper-server"))
    if not os.path.exists(whisper_bin_path):
        fallback_path = os.path.abspath(os.path.join(whisper_dir, "build", "whisper-server"))
        if os.path.exists(fallback_path):
            whisper_bin_path = fallback_path

    config_path = os.path.join(HERD_HOME, "config.json")
    config_data = {
        "LLAMA_SERVER_BIN": llama_bin_path,
        "WHISPER_SERVER_BIN": whisper_bin_path
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    console.print("\n[bold green]Herd setup completed successfully![/bold green]")
    console.print(f"Custom binary paths registered in [bold cyan]{config_path}[/bold cyan]:")
    console.print(f"  llama-server: [bold white]{llama_bin_path}[/bold white]")
    console.print(f"  whisper-server: [bold white]{whisper_bin_path}[/bold white]")


def check_cpu_flags() -> dict:
    """Detects CPU instruction capabilities on Linux and macOS."""
    flags = {"avx2": False, "avx512": False, "neon": False}
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read().lower()
                flags["avx2"] = "avx2" in content
                flags["avx512"] = "avx512" in content or "avx-512" in content
                flags["neon"] = "neon" in content or "asimd" in content
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            res = subprocess.run(["sysctl", "-a"], capture_output=True, text=True)
            content = res.stdout.lower()
            flags["avx2"] = "hw.optional.avx2: 1" in content or "avx2" in content
            flags["neon"] = "hw.optional.neon: 1" in content or "neon" in content
        except Exception:
            pass
    return flags


def check_gpu_info() -> Optional[dict]:
    """Retrieves NVIDIA GPU model and VRAM size if nvidia-smi is available."""
    nv_smi = shutil.which("nvidia-smi")
    if not nv_smi:
        return None
    try:
        res = subprocess.run(
            [nv_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        parts = res.stdout.strip().split(",")
        if len(parts) >= 3:
            mem_mb = float(parts[1].strip())
            return {
                "name": parts[0].strip(),
                "vram_gb": round(mem_mb / 1024.0, 2),
                "driver": parts[2].strip()
            }
    except Exception:
        pass
    return None


def get_tool_version(binary_name: str, version_arg: str = "--version") -> str:
    """Checks if a development tool is installed and retrieves its version string."""
    path = shutil.which(binary_name)
    if not path:
        return "[red]Not Installed[/red]"
    try:
        res = subprocess.run([path, version_arg], capture_output=True, text=True)
        first_line = res.stdout.strip().split("\n")[0]
        return f"[green]Installed[/green] ({first_line})"
    except Exception:
        return "[green]Installed[/green] (unknown version)"


@app.command(name="doctor")
def doctor():
    """Runs a system-wide hardware, prerequisite, and gateway state diagnostic."""
    import platform
    from rich.panel import Panel
    from rich.console import Group

    console.print("\n🩺 [bold green]Running Herd Doctor Diagnosis...[/bold green]\n")

    # 1. System & CPU
    sys_os = platform.system()
    sys_release = platform.release()
    sys_arch = platform.machine()
    py_ver = platform.python_version()
    
    cpu_flags = check_cpu_flags()
    def fmt_flag(supported: bool) -> str:
        return "[green]Yes[/green]" if supported else "[red]No[/red]"

    cpu_section = (
        f"[bold white]System Info:[/bold white]\n"
        f"  OS: {sys_os} ({sys_release})\n"
        f"  Architecture: {sys_arch}\n"
        f"  Python Version: {py_ver}\n"
        f"  CPU Features: AVX2: {fmt_flag(cpu_flags['avx2'])}, AVX512: {fmt_flag(cpu_flags['avx512'])}, Neon: {fmt_flag(cpu_flags['neon'])}\n"
    )

    # 2. GPU Check
    gpu = check_gpu_info()
    if gpu:
        gpu_section = (
            f"[bold white]GPU Hardware (NVIDIA):[/bold white]\n"
            f"  Device Name: [cyan]{gpu['name']}[/cyan]\n"
            f"  VRAM Available: [green]{gpu['vram_gb']:.2f} GB[/green]\n"
            f"  Driver Version: {gpu['driver']}\n"
        )
    else:
        gpu_section = (
            "[bold white]GPU Hardware:[/bold white]\n"
            "  Device Name: [yellow]No NVIDIA GPU detected[/yellow] (or nvidia-smi not in PATH)\n"
        )

    # 3. Development Prereqs
    git_status = get_tool_version("git")
    cmake_status = get_tool_version("cmake")
    compiler_status = get_tool_version("g++") if sys_os != "Darwin" else get_tool_version("clang")
    compiler_lbl = "G++" if sys_os != "Darwin" else "Clang"

    prereq_section = (
        f"[bold white]Prerequisites (for compilation):[/bold white]\n"
        f"  Git: {git_status}\n"
        f"  CMake: {cmake_status}\n"
        f"  {compiler_lbl}: {compiler_status}\n"
    )

    # 4. Gateway Status
    active_info = ""
    if is_gateway_running():
        host = HERD_HOST
        if host == "0.0.0.0":
            host = "127.0.0.1"
        try:
            res = httpx.get(f"http://{host}:{HERD_PORT}/v1/models/active")
            active_models = res.json()
            active_info = f"[green]Running[/green] ({len(active_models)} active model{'s' if len(active_models) != 1 else ''})"
        except Exception:
            active_info = "[green]Running[/green]"
    else:
        active_info = "[red]Not Running[/red]"

    def fmt_bin(path: Optional[str]) -> str:
        if path and os.path.exists(path):
            return f"[green]Found[/green] ({path})"
        return "[red]Not Found[/red] (will search PATH at startup)"

    gateway_section = (
        f"[bold white]Herd Gateway & Environment:[/bold white]\n"
        f"  Home Directory: {HERD_HOME}\n"
        f"  API Host / Port: {HERD_HOST}:{HERD_PORT}\n"
        f"  Idle Timeout: {IDLE_TIMEOUT}s\n"
        f"  Gateway Status: {active_info}\n"
        f"  llama-server: {fmt_bin(LLAMA_SERVER_BIN)}\n"
        f"  whisper-server: {fmt_bin(WHISPER_SERVER_BIN)}\n"
    )

    # Output formatting using a Panel
    diagnostic_info = Group(
        cpu_section,
        "\n",
        gpu_section,
        "\n",
        prereq_section,
        "\n",
        gateway_section
    )

    console.print(Panel(
        diagnostic_info,
        title="[bold green]Herd Doctor Diagnosis Report[/bold green]",
        border_style="green",
        expand=False
    ))


def main():
    app()


if __name__ == "__main__":
    main()
