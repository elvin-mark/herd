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
    load_config,
    save_config,
    DEFAULT_LLM,
    DEFAULT_EMBEDDING,
    DEFAULT_WHISPER,
)
from herd.services.downloader import (
    list_hf_repository_files,
    download_file,
    parse_model_identifier,
    resolve_model_path,
)
from herd.services.rag import index_directory, get_embedding, search_vectors

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


async def chat_interactive(model_name: str, context_model: Optional[str] = None):
    """Launches the CLI chat session loops."""
    console.print(
        f"\n[bold green]Chatting with {model_name} (Herd Gateway)[/bold green]"
    )
    console.print("Type [bold cyan]/help[/bold cyan] to see available commands. Press Ctrl+C to stop generation.\n")

    # Load embedding model if RAG context is active
    if context_model:
        url_load = f"http://127.0.0.1:{HERD_PORT}/v1/models/load"
        try:
            httpx.post(url_load, json={"model": context_model, "is_embedding": True}, timeout=45.0)
            console.print(f"[dim]RAG Active: Retrieving context from embedding model '{context_model}'[/dim]\n")
        except Exception as e:
            console.print(f"[red]Warning: Failed to load RAG embedding model: {e}[/red]")
            context_model = None

    messages = []
    while True:
        try:
            user_input = typer.prompt(">>> ", prompt_suffix="").strip()
            if not user_input:
                continue

            # Process slash commands
            if user_input.startswith("/"):
                cmd_parts = user_input.split(" ", 1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

                if cmd in ["/exit", "/quit"]:
                    break
                elif cmd == "/help":
                    console.print("\n[bold cyan]Available Chat Commands:[/bold cyan]")
                    console.print("  [bold white]/help[/bold white]               - Show this help menu.")
                    console.print("  [bold white]/clear[/bold white] or [bold white]/reset[/bold white]   - Clear the chat history.")
                    console.print("  [bold white]/system <prompt>[/bold white]     - Set or update the system prompt.")
                    console.print("  [bold white]/export [filename][/bold white]   - Export the chat history to a Markdown file.")
                    console.print("  [bold white]/exit[/bold white] or [bold white]/quit[/bold white]     - Exit the chat session.\n")
                    continue
                elif cmd in ["/clear", "/reset"]:
                    messages = []
                    console.print("[yellow]Chat history cleared.[/yellow]")
                    continue
                elif cmd == "/system":
                    if not arg:
                        console.print("[red]Usage: /system <prompt>[/red]")
                        continue
                    # Update or prepend system prompt
                    has_system = False
                    for idx, msg in enumerate(messages):
                        if msg["role"] == "system":
                            messages[idx]["content"] = arg
                            has_system = True
                            break
                    if not has_system:
                        messages.insert(0, {"role": "system", "content": arg})
                    console.print(f"[yellow]System prompt updated to:[/yellow] [italic]{arg}[/italic]")
                    continue
                elif cmd == "/export":
                    filename = arg if arg else "chat_export.md"
                    try:
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(f"# Chat Session with {model_name}\n\n")
                            for msg in messages:
                                role = msg["role"].capitalize()
                                f.write(f"### {role}\n{msg['content']}\n\n")
                        console.print(f"[green]Chat session exported to {filename}[/green]")
                    except Exception as e:
                        console.print(f"[red]Failed to export chat: {e}[/red]")
                    continue
                else:
                    console.print(f"[red]Unknown command: {cmd}. Type /help for assistance.[/red]")
                    continue

            # Retrieve context if RAG is active
            retrieved_context = ""
            if context_model:
                try:
                    query_vector = await get_embedding(user_input, context_model)
                    matches = search_vectors(query_vector, context_model, top_k=3)
                    if matches:
                        retrieved_context = "\n\n".join([
                            f"Source: {os.path.basename(m['file_path'])}\n{m['text']}"
                            for m in matches
                        ])
                except Exception:
                    pass

            messages.append({"role": "user", "content": user_input})
            print("Response: ", end="")

            try:
                payload_messages = messages.copy()
                if retrieved_context:
                    payload_messages[-1] = {
                        "role": "user",
                        "content": (
                            "Use the following context to answer the question.\n\n"
                            f"Context:\n{retrieved_context}\n\n"
                            f"Question: {user_input}"
                        )
                    }
                assistant_response = await stream_chat_completions(model_name, payload_messages)
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
    model_name: Optional[str] = typer.Argument(
        None,
        help="Model identifier to run. If omitted, resolves to active model, configured default, or first local model.",
    ),
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
    context_model: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Perform interactive RAG chat by semantic context retrieval from this embedding model. If 'auto', resolves to default_embedding config.",
    ),
):
    """Starts a model and opens an interactive chat (or runs transcription server)."""
    # 1. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # Resolve context model default if set to 'auto' or omitted but available
    if context_model == "auto" or (context_model is None and DEFAULT_EMBEDDING):
        context_model = DEFAULT_EMBEDDING

    # Resolve model
    chosen_model = model_name
    if not chosen_model:
        if whisper:
            chosen_model = DEFAULT_WHISPER
            if not chosen_model:
                models = get_local_models_info()
                w_models = [m["name"] for m in models if "whisper" in m["name"].lower()]
                if w_models:
                    chosen_model = w_models[0]
        elif embedding:
            chosen_model = DEFAULT_EMBEDDING
            if not chosen_model:
                models = get_local_models_info()
                emb_models = [m["name"] for m in models if "embedding" in m["name"].lower() or "bert" in m["name"].lower()]
                if emb_models:
                    chosen_model = emb_models[0]
        else:
            chosen_model = find_running_llm()

    if not chosen_model:
        console.print("[red]Error: No model name specified and no suitable default model configured.[/red]")
        console.print("Please pull a model first or configure defaults using [bold cyan]herd config set[/bold cyan].")
        raise typer.Exit(1)

    model_name = chosen_model

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
        asyncio.run(chat_interactive(model_name, context_model))


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
def stop(
    model_name: Optional[str] = typer.Argument(None, help="Model identifier to stop. Required unless --all is specified."),
    stop_all: bool = typer.Option(False, "--all", "-a", help="Stop all running model processes."),
):
    """Stops a running model process."""
    if not is_gateway_running():
        console.print("[yellow]Herd API gateway is not running.[/yellow]")
        return

    url = f"http://127.0.0.1:{HERD_PORT}/v1/models/unload"

    if stop_all:
        # Fetch active models
        active_url = f"http://127.0.0.1:{HERD_PORT}/v1/models/active"
        try:
            active_res = httpx.get(active_url, timeout=5.0)
            if active_res.status_code != 200:
                console.print(f"[red]Failed to query active models: {active_res.text}[/red]")
                raise typer.Exit(1)
            active_models = active_res.json()
        except Exception as e:
            console.print(f"[red]Error fetching active models: {e}[/red]")
            raise typer.Exit(1)

        if not active_models:
            console.print("[yellow]No active running models found.[/yellow]")
            return

        for m in active_models:
            m_name = m["model"]
            try:
                response = httpx.post(url, json={"model": m_name})
                if response.status_code == 200:
                    console.print(f"[green]Successfully stopped model '{m_name}'.[/green]")
                else:
                    console.print(f"[red]Failed to stop model '{m_name}': {response.text}[/red]")
            except Exception as e:
                console.print(f"[red]Error stopping model '{m_name}': {e}[/red]")
    else:
        if not model_name:
            console.print("[red]Error: Please specify a model name, or use --all (-a) to stop all running models.[/red]")
            raise typer.Exit(1)

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


DEFAULT_BENCHMARK_PROMPTS = [
    "Explain the concept of quantum computing in one simple sentence.",
    "Write a short, engaging story about a time traveler who gets stuck in the year 1999.",
    "Draft a detailed step-by-step guide for deploying a Dockerized Python FastAPI web application to production."
]


async def run_benchmark_async(model_name: str, custom_prompts: Optional[list[str]], rounds: int):
    # 1. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # 2. Check if model exists locally
    try:
        resolve_model_path(model_name)
    except FileNotFoundError:
        console.print(f"[red]Error: Model '{model_name}' not found locally. Please pull it first.[/red]")
        raise typer.Exit(1)

    # 3. Load model in gateway
    console.print(f"Loading [bold cyan]{model_name}[/bold cyan] and running benchmark suite ({rounds} rounds per prompt)...")
    url_load = f"http://127.0.0.1:{HERD_PORT}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": model_name}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to pre-load model: {e}[/red]")
        raise typer.Exit(1)

    prompts = custom_prompts if custom_prompts else DEFAULT_BENCHMARK_PROMPTS
    results = []

    for idx, prompt in enumerate(prompts):
        console.print(f"\n[bold magenta]Prompt {idx + 1}/{len(prompts)}:[/bold magenta] [italic]\"{prompt[:60]}...\"[/italic]")

        ttfts = []
        speeds = []
        mems = []
        cpus = []

        for r in range(rounds):
            print(f"  Round {r + 1}/{rounds}...", end="", flush=True)

            url_chat = f"http://127.0.0.1:{HERD_PORT}/v1/chat/completions"
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }

            start_time = time.time()
            first_token_time = None
            token_count = 0

            # Get resource stats concurrently during inference
            async def get_stats_during_inference():
                await asyncio.sleep(0.5)  # Wait for it to start processing
                host = HERD_HOST
                if host == "0.0.0.0":
                    host = "127.0.0.1"
                try:
                    async with httpx.AsyncClient() as client:
                        res = await client.get(f"http://{host}:{HERD_PORT}/v1/models/active")
                        active = res.json()
                        for m in active:
                            if m["model"] == model_name:
                                return m.get("memory_bytes", 0), m.get("cpu_percent", 0.0)
                except Exception:
                    pass
                return 0, 0.0

            stats_task = asyncio.create_task(get_stats_during_inference())

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url_chat, json=payload) as response:
                        async for line in response.aiter_lines():
                            if not line:
                                continue
                            if first_token_time is None:
                                first_token_time = time.time()
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
                                            token_count += 1
                                except Exception:
                                    pass
            except Exception as e:
                console.print(f" [red]Failed: {e}[/red]")
                continue

            end_time = time.time()
            duration = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else duration
            speed = token_count / (end_time - first_token_time) if (first_token_time and end_time > first_token_time) else 0.0

            mem_bytes, cpu_pct = await stats_task

            ttfts.append(ttft)
            speeds.append(speed)
            if mem_bytes > 0:
                mems.append(mem_bytes)
            if cpu_pct > 0.0:
                cpus.append(cpu_pct)

            console.print(f" Done ({speed:.1f} tok/sec, TTFT: {ttft:.2f}s)")

        if ttfts and speeds:
            avg_ttft = sum(ttfts) / len(ttfts)
            avg_speed = sum(speeds) / len(speeds)
            avg_mem = sum(mems) / len(mems) if mems else 0
            avg_cpu = sum(cpus) / len(cpus) if cpus else 0.0

            mem_gb = avg_mem / (1024 * 1024 * 1024)
            mem_str = f"{mem_gb:.2f} GB" if mem_gb >= 0.1 else f"{avg_mem / (1024 * 1024):.1f} MB"

            results.append({
                "prompt": prompt[:40] + "...",
                "ttft": avg_ttft,
                "speed": avg_speed,
                "memory": mem_str if avg_mem > 0 else "N/A",
                "cpu": f"{avg_cpu:.1f}%" if avg_cpu > 0.0 else "N/A"
            })

    # Display results
    if results:
        table = Table(title=f"Benchmark Results: {model_name}")
        table.add_column("Prompt Sample", style="cyan")
        table.add_column("Avg TTFT (s)", style="yellow")
        table.add_column("Avg Speed (tok/sec)", style="bold green")
        table.add_column("Peak Memory", style="magenta")
        table.add_column("Avg CPU %", style="blue")

        for r in results:
            table.add_row(
                r["prompt"],
                f"{r['ttft']:.3f}s",
                f"{r['speed']:.1f} tok/s",
                r["memory"],
                r["cpu"]
            )

        console.print("\n")
        console.print(table)


@app.command(name="benchmark")
def benchmark(
    model_name: str = typer.Argument(..., help="Model identifier to benchmark."),
    prompts: Optional[str] = typer.Option(
        None,
        "--prompts",
        "-p",
        help="Comma-separated custom prompts to test. If not specified, uses built-in prompts.",
    ),
    rounds: int = typer.Option(
        3,
        "--rounds",
        "-r",
        help="Number of evaluation rounds to run per prompt to calculate average statistics.",
    ),
):
    """Benchmarks a model's load time, prompt ingestion latency (TTFT), generation speed, and system memory footprint."""
    custom_list = [p.strip() for p in prompts.split(",")] if prompts else None
    asyncio.run(run_benchmark_async(model_name, custom_list, rounds))


@app.command(name="suggest")
def suggest():
    """Analyzes system hardware (RAM and VRAM) and suggests compatible LLMs and Whisper models."""
    import psutil
    from rich.panel import Panel

    console.print("\n🔍 [bold green]Auditing hardware to generate model recommendations...[/bold green]\n")

    # Check RAM
    try:
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024 * 1024 * 1024)
    except Exception:
        ram_gb = 8.0  # Fallback default

    # Check GPU
    gpu = check_gpu_info()
    vram_gb = gpu["vram_gb"] if gpu else 0.0

    # CPU recommendations based on RAM
    if ram_gb >= 16.0:
        cpu_llm = "unsloth/Qwen3.5-7B-Instruct-GGUF:Q4_K_M"
        cpu_desc = "Runs comfortably on CPU. Balanced speed/reasoning."
    elif ram_gb >= 8.0:
        cpu_llm = "unsloth/Qwen3.5-3B-Instruct-GGUF:Q4_K_M"
        cpu_desc = "Optimal size for standard CPU memory. Good code/chat."
    else:
        cpu_llm = "Qwen/Qwen3.5-0.8B:Q8_0"
        cpu_desc = "Lightweight model to prevent memory swapping on low RAM."

    # GPU recommendations based on VRAM
    gpu_llm = None
    gpu_desc = ""
    if vram_gb >= 16.0:
        gpu_llm = "unsloth/Qwen3.5-14B-Instruct-GGUF:Q8_0"
        gpu_desc = "Fits fully in VRAM. Outstanding coding/reasoning speed."
    elif vram_gb >= 8.0:
        gpu_llm = "unsloth/Llama-3-8B-Instruct-GGUF:Q8_0"
        gpu_desc = "Fits fully in VRAM. Great generalist assistant at extreme speeds."
    elif vram_gb >= 4.0:
        gpu_llm = "unsloth/Qwen3.5-3B-Instruct-GGUF:Q8_0"
        gpu_desc = "Fits in low VRAM. Good chat response speeds."

    # Whisper recommendations (universal)
    whisper_rec = "ggerganov/whisper.cpp:ggml-base.en.bin"
    whisper_desc = "Lightweight English-only model. Transcribes near real-time."
    whisper_multilingual = "ggerganov/whisper.cpp:ggml-small.bin"
    whisper_multi_desc = "Great for transcribing Spanish, French, and 90+ other languages."

    # Build the report string
    report = []
    report.append("[bold white]Detected Hardware:[/bold white]")
    report.append(f"  System RAM: [cyan]{ram_gb:.1f} GB[/cyan]")
    if gpu:
        report.append(f"  GPU Name: [cyan]{gpu['name']}[/cyan]")
        report.append(f"  GPU VRAM: [green]{vram_gb:.2f} GB[/green]")
    else:
        report.append("  GPU Name: [yellow]No NVIDIA GPU detected[/yellow] (CPU execution only)")
    report.append("")

    report.append("[bold green]💻 Recommended LLM (CPU Mode):[/bold green]")
    report.append(f"  Model: [white]{cpu_llm}[/white]")
    report.append(f"  Details: {cpu_desc}")
    report.append("  Run Command: [bold cyan]herd run " + cpu_llm + "[/bold cyan]")
    report.append("")

    if gpu_llm:
        report.append("[bold green]🎮 Recommended LLM (GPU Accelerated Mode):[/bold green]")
        report.append(f"  Model: [white]{gpu_llm}[/white]")
        report.append(f"  Details: {gpu_desc}")
        report.append("  Run Command: [bold cyan]herd run " + gpu_llm + "[/bold cyan]")
        report.append("")

    report.append("[bold green]🎙️ Recommended Speech-to-Text (Whisper):[/bold green]")
    report.append(f"  English: [white]{whisper_rec}[/white] ({whisper_desc})")
    report.append(f"  Multilingual: [white]{whisper_multilingual}[/white] ({whisper_multi_desc})")
    report.append("  Pull Command: [bold cyan]herd pull " + whisper_rec + "[/bold cyan]")

    console.print(Panel(
        "\n".join(report),
        title="[bold green]Herd Model Recommendation Report[/bold green]",
        border_style="green",
        expand=False
    ))


def ms_to_srt_time(ms: int) -> str:
    """Converts milliseconds to SRT time format HH:MM:SS,mmm"""
    secs, msecs = divmod(ms, 1000)
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"


def ms_to_vtt_time(ms: int) -> str:
    """Converts milliseconds to VTT time format HH:MM:SS.mmm"""
    secs, msecs = divmod(ms, 1000)
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"


@app.command(name="transcribe")
def transcribe(
    audio_file: str = typer.Argument(..., help="Path to the local audio file to transcribe."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model identifier. If not specified, auto-selects the first locally downloaded Whisper model.",
    ),
    output_format: str = typer.Option(
        "txt",
        "--format",
        "-f",
        help="Output transcription format (txt, srt, vtt).",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the output transcription. Defaults to <audio_file_name>.<format>.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Target language code (e.g. en, es, fr). If not set, auto-detects language.",
    ),
):
    """Transcribes an audio file into text, subtitle formats (SRT/VTT), or raw text using Whisper."""
    # 1. Ensure audio file exists
    if not os.path.exists(audio_file):
        console.print(f"[red]Error: Audio file not found at: {audio_file}[/red]")
        raise typer.Exit(1)

    # 2. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # 3. Resolve Whisper model to use
    chosen_model = model_name
    if not chosen_model:
        if DEFAULT_WHISPER:
            chosen_model = DEFAULT_WHISPER
        else:
            whisper_models = [
                m["name"]
                for m in get_local_models_info()
                if "whisper" in m["name"].lower() or m["filename"].endswith(".bin")
            ]
            if whisper_models:
                chosen_model = whisper_models[0]
                console.print(f"[yellow]No model specified. Auto-selected local Whisper model: [bold]{chosen_model}[/bold][/yellow]")

    if not chosen_model:
        console.print("[red]Error: No Whisper models found locally and no default Whisper model configured.[/red]")
        console.print("Please download a Whisper model first: [bold cyan]herd pull ggerganov/whisper.cpp:ggml-base.en.bin[/bold cyan]")
        raise typer.Exit(1)

    # 4. Resolve output path
    fmt = output_format.lower()
    if fmt not in ["txt", "srt", "vtt"]:
        console.print(f"[red]Error: Unsupported output format '{output_format}'. Choose from: txt, srt, vtt.[/red]")
        raise typer.Exit(1)

    if not output_file:
        base, _ = os.path.splitext(audio_file)
        dest_path = f"{base}.{fmt}"
    else:
        dest_path = output_file

    console.print(f"Loading Whisper model [bold cyan]{chosen_model}[/bold cyan] in Gateway...")

    # 5. Send transcription request to the Gateway
    url = f"http://127.0.0.1:{HERD_PORT}/v1/audio/transcriptions"

    # Open and stream file
    try:
        with open(audio_file, "rb") as f_bin:
            files = {"file": (os.path.basename(audio_file), f_bin, "audio/wav")}
            data = {
                "model": chosen_model,
                "response_format": "json"
            }
            if language:
                data["language"] = language

            console.print("[bold green]Transcribing audio file...[/bold green] (this may take a few moments)")
            response = httpx.post(url, files=files, data=data, timeout=None)
    except Exception as e:
        console.print(f"[red]Error contacting Gateway transcription server: {e}[/red]")
        raise typer.Exit(1)

    if response.status_code != 200:
        console.print(f"[red]Transcription failed: {response.text}[/red]")
        raise typer.Exit(1)

    result = response.json()

    # 6. Format and save the transcription
    segments = result.get("segments", [])

    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            if fmt == "txt":
                f.write(result.get("text", "").strip())
            elif fmt == "srt":
                for idx, seg in enumerate(segments):
                    from_ms = seg.get("offsets", {}).get("from", 0)
                    to_ms = seg.get("offsets", {}).get("to", 0)
                    start_str = ms_to_srt_time(from_ms)
                    end_str = ms_to_srt_time(to_ms)
                    text = seg.get("text", "").strip()
                    f.write(f"{idx + 1}\n{start_str} --> {end_str}\n{text}\n\n")
            elif fmt == "vtt":
                f.write("WEBVTT\n\n")
                for idx, seg in enumerate(segments):
                    from_ms = seg.get("offsets", {}).get("from", 0)
                    to_ms = seg.get("offsets", {}).get("to", 0)
                    start_str = ms_to_vtt_time(from_ms)
                    end_str = ms_to_vtt_time(to_ms)
                    text = seg.get("text", "").strip()
                    f.write(f"{start_str} --> {end_str}\n{text}\n\n")

        console.print(f"\n[bold green]Success![/bold green] Transcription saved to: [bold cyan]{dest_path}[/bold cyan]")
    except Exception as e:
        console.print(f"[red]Error writing transcription file: {e}[/red]")
        raise typer.Exit(1)


def get_local_ip() -> str:
    """Finds the primary local IP address of this machine."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


@app.command(name="share")
def share(
    qr: bool = typer.Option(
        False,
        "--qr",
        "-q",
        help="Generate an ASCII QR code in the terminal for easy mobile pairing.",
    ),
    public: bool = typer.Option(
        False,
        "--public",
        "-p",
        help="Expose the gateway to the public internet using a free Cloudflare Tunnel.",
    ),
):
    """Exposes connection strings and generates pairing helper for local network or public devices."""
    port = HERD_PORT

    if public:
        cloudflared_bin = shutil.which("cloudflared")
        if not cloudflared_bin:
            console.print("[red]Error: 'cloudflared' is not installed or not in PATH.[/red]")
            console.print("Please install Cloudflare Tunnel first. Examples:")
            console.print("  [bold white]macOS:[/bold white] brew install cloudflared")
            console.print("  [bold white]Linux:[/bold white] sudo apt install cloudflared")
            raise typer.Exit(1)

        console.print("[bold cyan]Starting public Cloudflare Tunnel...[/bold cyan]")
        try:
            process = subprocess.Popen(
                [cloudflared_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True
            )

            # Read lines to find the trycloudflare URL
            public_url = None
            start_time = time.time()
            while time.time() - start_time < 15.0:  # 15s timeout
                line = process.stdout.readline()
                if not line:
                    break
                import re
                match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
                if match:
                    public_url = match.group(1)
                    break

            if not public_url:
                console.print("[red]Error: Failed to retrieve Cloudflare Tunnel URL. Check if cloudflared is working correctly.[/red]")
                process.terminate()
                process.wait()
                raise typer.Exit(1)

            console.print("\n🌎 [bold green]Public Exposure Active![/bold green]\n")
            console.print(f"  Public API Base URL:  [bold cyan]{public_url}/v1[/bold cyan]")
            console.print(f"  Public Web Dashboard: [bold cyan]{public_url}[/bold cyan]")
            console.print("")
            console.print("[yellow]Your local Herd gateway is now securely accessible from anywhere in the world![/yellow]")

            if qr:
                try:
                    import qrcode
                    console.print("\n[bold yellow]Scan this QR Code to copy the Public API URL on your mobile device:[/bold yellow]\n")
                    qr_obj = qrcode.QRCode()
                    qr_obj.add_data(f"{public_url}/v1")
                    qr_obj.make()
                    qr_obj.print_ascii(tty=True)
                    console.print("")
                except ImportError:
                    pass

            console.print("[bold yellow]--- Press Ctrl+C to stop the tunnel and revoke the public URL ---[/bold yellow]\n")

            # Block and keep reading to keep process alive, print errors if any
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # Silently consume output to avoid terminal clutter, but keep loop alive
                pass

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping Cloudflare Tunnel...[/yellow]")
        finally:
            try:
                import signal
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait()
            except Exception:
                try:
                    process.terminate()
                    process.wait()
                except Exception:
                    pass
            console.print("[green]Public URL revoked successfully.[/green]")
        return

    # Default local share logic
    ip = get_local_ip()
    url = f"http://{ip}:{port}/v1"

    console.print("\n📶 [bold green]Herd Connection & Exposer Helper[/bold green]\n")
    console.print("Your Gateway is accessible on the local network at:")
    console.print(f"  API Base URL:  [bold cyan]{url}[/bold cyan]")
    console.print(f"  Web Dashboard: [bold cyan]http://{ip}:{port}[/bold cyan]")
    console.print("")
    console.print("Configure your mobile client (e.g. Chatbox, LibreChat) with this API Base URL.")
    console.print("")

    if qr:
        try:
            import qrcode
            console.print("[bold yellow]Scan this QR Code to copy the API Base URL on your mobile device:[/bold yellow]\n")
            qr_obj = qrcode.QRCode()
            qr_obj.add_data(url)
            qr_obj.make()
            qr_obj.print_ascii(tty=True)
            console.print("")
        except ImportError:
            console.print("[yellow]Notice: 'qrcode' package is not installed. To display QR codes, install it via:[/yellow]")
            console.print("  [bold cyan]pip install qrcode[/bold cyan]")
            console.print("")


def extract_gguf_metadata(file_path: str) -> dict:
    """Heuristically extracts basic metadata (name, architecture) from the GGUF binary header."""
    meta = {"architecture": "Unknown", "name": "Unknown"}
    try:
        with open(file_path, "rb") as f:
            # Read first 128KB which contains the KV metadata header
            header = f.read(128 * 1024)

            # Check magic bytes 'GGUF'
            if header[:4] != b"GGUF":
                return meta

            # Parse general.architecture
            arch_idx = header.find(b"general.architecture")
            if arch_idx != -1:
                # "general.architecture" length is 20
                window = header[arch_idx + 20 : arch_idx + 20 + 100]
                if len(window) >= 12:
                    val_type = int.from_bytes(window[:4], "little")
                    if val_type == 8:  # GGUF String type
                        str_len = int.from_bytes(window[4:12], "little")
                        if 0 < str_len < 100 and len(window) >= 12 + str_len:
                            arch_bytes = window[12 : 12 + str_len]
                            meta["architecture"] = arch_bytes.decode("utf-8", errors="ignore").strip().capitalize()

            # Parse general.name
            name_idx = header.find(b"general.name")
            if name_idx != -1:
                # "general.name" length is 12
                window = header[name_idx + 12 : name_idx + 12 + 100]
                if len(window) >= 12:
                    val_type = int.from_bytes(window[:4], "little")
                    if val_type == 8:  # GGUF String type
                        str_len = int.from_bytes(window[4:12], "little")
                        if 0 < str_len < 100 and len(window) >= 12 + str_len:
                            name_bytes = window[12 : 12 + str_len]
                            meta["name"] = name_bytes.decode("utf-8", errors="ignore").strip()
    except Exception:
        pass
    return meta


@app.command(name="show")
def show(
    model_name: str = typer.Argument(..., help="Model identifier to view details for.")
):
    """Displays detailed metadata, file paths, size, and architecture details for a model."""
    try:
        model_path = resolve_model_path(model_name)
    except FileNotFoundError:
        console.print(f"[red]Error: Model '{model_name}' not found locally.[/red]")
        raise typer.Exit(1)

    import time
    from rich.panel import Panel

    file_size = os.path.getsize(model_path)
    size_gb = file_size / (1024 * 1024 * 1024)
    size_str = f"{size_gb:.2f} GB" if size_gb >= 1.0 else f"{file_size / (1024 * 1024):.2f} MB"

    mtime = os.path.getmtime(model_path)
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))

    # Extract GGUF header metadata
    gguf_meta = extract_gguf_metadata(model_path)

    # Estimate Quantization from filename
    filename = os.path.basename(model_path)
    import re
    quant_match = re.search(r'([qI]?[0-9]_[A-Za-z0-9_]+)', filename, re.IGNORECASE)
    quant = quant_match.group(1).upper() if quant_match else "Unknown / F16"

    # Print Panel
    info = []
    info.append("[bold white]Model Details:[/bold white]")
    info.append(f"  Identifier:   [cyan]{model_name}[/cyan]")
    info.append(f"  GGUF Name:    {gguf_meta['name']}")
    info.append(f"  Architecture: [magenta]{gguf_meta['architecture']}[/magenta]")
    info.append(f"  Quantization: [green]{quant}[/green]")
    info.append("")
    info.append("[bold white]File Information:[/bold white]")
    info.append(f"  Filename:     {filename}")
    info.append(f"  Path:         {model_path}")
    info.append(f"  Size:         {size_str}")
    info.append(f"  Last Modified: {time_str}")

    console.print(Panel(
        "\n".join(info),
        title=f"[bold green]Model Inspector: {model_name}[/bold green]",
        border_style="green",
        expand=False
    ))


@app.command(name="clean")
def clean(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Clean logs immediately without prompting for confirmation.",
    ),
):
    """Cleans up inactive model logs in the log directory to free up disk space."""
    if not os.path.exists(HERD_LOGS_DIR):
        console.print("[yellow]Log directory does not exist. Nothing to clean.[/yellow]")
        return

    # Find active logs to preserve
    active_log_names = {"gateway.log"}
    if is_gateway_running():
        host = HERD_HOST
        if host == "0.0.0.0":
            host = "127.0.0.1"
        try:
            res = httpx.get(f"http://{host}:{HERD_PORT}/v1/models/active")
            active = res.json()
            for m in active:
                model_safe = m["model"].replace("/", "_").replace(":", "_")
                active_log_names.add(f"{model_safe}.log")
        except Exception:
            pass

    log_files = [f for f in os.listdir(HERD_LOGS_DIR) if f.endswith(".log")]
    to_delete = [f for f in log_files if f not in active_log_names]

    if not to_delete:
        console.print("[green]No inactive logs found. Your log directory is clean![/green]")
        return

    # Calculate total size
    total_bytes = 0
    for f_name in to_delete:
        total_bytes += os.path.getsize(os.path.join(HERD_LOGS_DIR, f_name))

    size_str = (
        f"{total_bytes / (1024 * 1024):.2f} MB"
        if total_bytes >= 1024 * 1024
        else f"{total_bytes / 1024:.2f} KB"
    )

    console.print(f"Found {len(to_delete)} inactive log file(s) ({size_str}).")

    if not force:
        confirm = typer.confirm("Are you sure you want to delete these log files?")
        if not confirm:
            console.print("[yellow]Cleanup aborted.[/yellow]")
            return

    deleted_count = 0
    for f_name in to_delete:
        try:
            os.remove(os.path.join(HERD_LOGS_DIR, f_name))
            deleted_count += 1
        except Exception as e:
            console.print(f"[red]Failed to delete {f_name}: {e}[/red]")

    console.print(f"[green]Successfully deleted {deleted_count} inactive log file(s).[/green]")


@app.command(name="index")
def index(
    directory: str = typer.Argument(..., help="Path to the local directory to index."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The embedding model identifier to use. If not specified, uses default_embedding config.",
    ),
):
    """Recursively chunks and embeds files in a directory, storing them in the local database."""
    if not os.path.exists(directory):
        console.print(f"[red]Error: Directory does not exist: {directory}[/red]")
        raise typer.Exit(1)

    # 1. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    chosen_model = model_name if model_name else DEFAULT_EMBEDDING
    if not chosen_model:
        models = get_local_models_info()
        emb_models = [m["name"] for m in models if "embedding" in m["name"].lower() or "bert" in m["name"].lower()]
        if emb_models:
            chosen_model = emb_models[0]

    if not chosen_model:
        console.print("[red]Error: No embedding model specified and no default embedding model configured.[/red]")
        raise typer.Exit(1)

    model_name = chosen_model

    # 2. Pre-load the embedding model
    console.print(f"Ensuring embedding model [bold magenta]{model_name}[/bold magenta] is loaded...")
    url_load = f"http://127.0.0.1:{HERD_PORT}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": model_name, "is_embedding": True}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load embedding model: {e}[/red]")
        raise typer.Exit(1)

    # 3. Perform indexing
    console.print(f"Indexing directory [bold cyan]{directory}[/bold cyan]...")
    try:
        count = asyncio.run(index_directory(directory, model_name))
        console.print(f"[bold green]Success![/bold green] Indexed {count} text chunks in the database.")
    except Exception as e:
        console.print(f"[red]Failed to index directory: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="ask")
def ask(
    query: str = typer.Argument(..., help="The question to ask the model using indexed context."),
    model_name: Optional[str] = typer.Argument(
        None,
        help="LLM model identifier to ask. If omitted, uses active or default LLM.",
    ),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The embedding model identifier to query context from. If omitted, uses default_embedding config.",
    ),
):
    """Semantic query: retrieves relevant indexed chunks and answers your question using the LLM."""
    # 1. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # Resolve LLM model
    chosen_llm = model_name if model_name else find_running_llm()
    if not chosen_llm:
        console.print("[red]Error: No LLM model specified and no default LLM configured.[/red]")
        raise typer.Exit(1)

    # Resolve embedding model
    chosen_emb = embedding_model if embedding_model else DEFAULT_EMBEDDING
    if not chosen_emb:
        models = get_local_models_info()
        emb_models = [m["name"] for m in models if "embedding" in m["name"].lower() or "bert" in m["name"].lower()]
        if emb_models:
            chosen_emb = emb_models[0]

    if not chosen_emb:
        console.print("[red]Error: No embedding model specified and no default embedding model configured.[/red]")
        raise typer.Exit(1)

    # 2. Pre-load the models
    url_load = f"http://127.0.0.1:{HERD_PORT}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_emb, "is_embedding": True}, timeout=45.0)
        httpx.post(url_load, json={"model": chosen_llm}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load models: {e}[/red]")
        raise typer.Exit(1)

    # 3. Retrieve context
    console.print("Searching semantic index for context...")
    try:
        query_vector = asyncio.run(get_embedding(query, chosen_emb))
        matches = search_vectors(query_vector, chosen_emb, top_k=5)
    except Exception as e:
        console.print(f"[red]Failed to query embeddings: {e}[/red]")
        raise typer.Exit(1)

    if not matches:
        console.print("[yellow]Warning: No context found in index for this embedding model. Answering without custom context.[/yellow]")
        context = ""
    else:
        console.print("\n[bold white]Retrieved Context Sources:[/bold white]")
        for idx, m in enumerate(matches):
            basename = os.path.basename(m["file_path"])
            console.print(f"  [{idx + 1}] {basename} (similarity: {m['similarity']:.3f})")
        context = "\n\n".join([
            f"Source: {os.path.basename(m['file_path'])}\nContent:\n{m['text']}"
            for m in matches
        ])

    # 4. Prompt construction
    system_prompt = (
        "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
        "If you do not know the answer, say so.\n\n"
        f"Context:\n{context}"
    )

    # 5. Stream response
    async def ask_async():
        url_chat = f"http://127.0.0.1:{HERD_PORT}/v1/chat/completions"
        payload = {
            "model": chosen_llm,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": True
        }
        console.print("\n[bold green]Answer:[/bold green]")
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url_chat, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content", "")
                            print(content, end="", flush=True)
                        except Exception:
                            pass
        print("\n")

    try:
        asyncio.run(ask_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")


def find_llama_quantize() -> Optional[str]:
    """Resolves the path to the llama-quantize binary, looking next to llama-server or on system PATH."""
    if LLAMA_SERVER_BIN and os.path.exists(LLAMA_SERVER_BIN):
        dir_path = os.path.dirname(LLAMA_SERVER_BIN)
        quant_bin = os.path.join(dir_path, "llama-quantize")
        if os.path.exists(quant_bin):
            return quant_bin
    return shutil.which("llama-quantize")


@app.command(name="search")
def search(
    query: str = typer.Argument(..., help="Search term for GGUF models on Hugging Face Hub."),
    limit: int = typer.Option(
        10,
        "--limit",
        "-l",
        help="Maximum number of search results to display.",
    ),
):
    """Searches Hugging Face Hub for GGUF model repositories matching the query."""
    url = f"https://huggingface.co/api/models?search={query}&filter=gguf&sort=downloads&direction=-1&limit={limit}"
    console.print(f"Searching Hugging Face Hub for GGUF models matching '[bold cyan]{query}[/bold cyan]'...")
    try:
        response = httpx.get(url, timeout=15.0)
        if response.status_code != 200:
            console.print(f"[red]Search failed: {response.text}[/red]")
            raise typer.Exit(1)
        results = response.json()
    except Exception as e:
        console.print(f"[red]Error connecting to Hugging Face Hub: {e}[/red]")
        raise typer.Exit(1)

    if not results:
        console.print("[yellow]No matching GGUF models found.[/yellow]")
        return

    table = Table(title=f"Hugging Face Search Results for '{query}'")
    table.add_column("Repository ID", style="cyan", no_wrap=True)
    table.add_column("Author", style="magenta")
    table.add_column("Downloads", style="green", justify="right")
    table.add_column("Likes", style="yellow", justify="right")

    for r in results:
        repo_id = r.get("id", "")
        author = r.get("author", "Unknown")
        downloads = r.get("downloads", 0)
        likes = r.get("likes", 0)

        # Format downloads count
        if downloads >= 1_000_000:
            dl_str = f"{downloads / 1_000_000:.1f}M"
        elif downloads >= 1_000:
            dl_str = f"{downloads / 1_000:.1f}k"
        else:
            dl_str = str(downloads)

        table.add_row(repo_id, author, dl_str, str(likes))

    console.print("\n")
    console.print(table)
    console.print("\nTo pull a model, use: [bold cyan]herd pull <repository_id>:<tag>[/bold cyan]\n")


@app.command(name="quantize")
def quantize(
    input_file: str = typer.Argument(..., help="Path to the source GGUF file (e.g. FP16/FP32)."),
    output_file: str = typer.Argument(..., help="Path to save the output quantized GGUF file."),
    method: str = typer.Argument(..., help="Quantization method (e.g. Q4_K_M, Q8_0, Q5_K_M)."),
):
    """Quantizes (compresses) a GGUF model file locally using the compiled llama-quantize binary."""
    if not os.path.exists(input_file):
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    quant_bin = find_llama_quantize()
    if not quant_bin:
        console.print("[red]Error: 'llama-quantize' binary not found.[/red]")
        console.print("Please make sure you have run [bold cyan]herd setup[/bold cyan] to build the compilation tools locally.")
        raise typer.Exit(1)

    console.print(f"Quantizing [bold cyan]{input_file}[/bold cyan] to [bold green]{output_file}[/bold green] using method [bold yellow]{method}[/bold yellow]...")
    try:
        cmd = [quant_bin, input_file, output_file, method]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
        process.wait()

        if process.returncode == 0:
            console.print(f"\n[bold green]Success![/bold green] Quantized model saved to {output_file}")
        else:
            console.print(f"\n[red]Quantization failed with exit code: {process.returncode}[/red]")
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Quantization interrupted.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error running quantization: {e}[/red]")
        raise typer.Exit(1)


@app.command(name="top")
def top():
    """Opens a real-time monitor displaying active model servers and resource utilization."""
    from rich.live import Live
    from rich.panel import Panel
    from rich.console import Group
    import time

    host = HERD_HOST
    if host == "0.0.0.0":
        host = "127.0.0.1"
    url = f"http://{host}:{HERD_PORT}/v1/models/active"

    def make_display() -> Panel:
        try:
            res = httpx.get(url, timeout=1.0)
            if res.status_code != 200:
                return Panel("[red]Error: Gateway returned unsuccessful status.[/red]", title="Herd Top")
            active = res.json()
        except Exception:
            return Panel("[yellow]Herd Gateway is not running.[/yellow]", title="Herd Top")

        if not active:
            return Panel("No models currently running.", title="Herd Top — Idle")

        table = Table()
        table.add_column("Model Server", style="cyan")
        table.add_column("Port", style="yellow")
        table.add_column("CPU %", style="green", width=25)
        table.add_column("Memory", style="magenta")
        table.add_column("Idle Time", style="blue")
        table.add_column("Mode", style="white")

        total_cpu = 0.0
        total_mem = 0

        for m in active:
            model_name = m["model"]
            port = m["port"]
            cpu = m.get("cpu_percent", 0.0)
            mem_bytes = m.get("memory_bytes", 0)
            mem_str = m.get("memory_str", "0 B")
            idle_sec = int(m.get("idle_seconds", 0))

            total_cpu += cpu
            total_mem += mem_bytes

            # Mode description
            if m.get("is_whisper"):
                mode = "Speech"
            elif m.get("is_embedding"):
                mode = "Embedding"
            else:
                mode = "Chat"

            # Formatting CPU bar
            cpu_bar_count = int(cpu / 5.0)  # Scale to max 20 blocks for 100%
            cpu_bar = "|" * min(cpu_bar_count, 20)
            cpu_color = "green" if cpu < 50.0 else ("yellow" if cpu < 80.0 else "red")
            cpu_display = f"[{cpu_color}]{cpu_bar:<20}[/] {cpu:.1f}%"

            # Format idle time to mm:ss
            mins, secs = divmod(idle_sec, 60)
            idle_str = f"{mins:02d}:{secs:02d}"

            table.add_row(model_name, str(port), cpu_display, mem_str, idle_str, mode)

        total_mem_gb = total_mem / (1024 * 1024 * 1024)
        total_mem_str = f"{total_mem_gb:.2f} GB" if total_mem_gb >= 0.1 else f"{total_mem / (1024 * 1024):.1f} MB"

        summary = (
            f"Active Models: [bold white]{len(active)}[/bold white] | "
            f"Total CPU: [bold white]{total_cpu:.1f}%[/bold white] | "
            f"Total Memory: [bold white]{total_mem_str}[/bold white]"
        )

        return Panel(
            Group(summary, "", table),
            title="[bold green]Herd Top — Real-Time Model Monitor[/bold green]",
            border_style="green",
            subtitle="[dim]Press Ctrl+C to exit[/dim]"
        )

    try:
        with Live(make_display(), refresh_per_second=2) as live:
            while True:
                time.sleep(0.5)
                live.update(make_display())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting monitor...[/yellow]")


db_app = typer.Typer(name="db", help="Manage the local RAG vector database index.")


@db_app.command(name="list")
def db_list():
    """Lists all files and paths currently indexed in the vector database."""
    from herd.services.rag import list_indexed_files
    try:
        rows = list_indexed_files()
    except Exception as e:
        console.print(f"[red]Error reading database: {e}[/red]")
        raise typer.Exit(1)

    if not rows:
        console.print("[yellow]The vector database index is empty.[/yellow]")
        return

    table = Table(title="Indexed Documents & Chunks")
    table.add_column("File / Directory Path", style="cyan")
    table.add_column("Embedding Model", style="magenta")
    table.add_column("Total Chunks", style="green", justify="right")

    for file_path, model_name, count in rows:
        table.add_row(file_path, model_name, str(count))

    console.print("\n")
    console.print(table)
    console.print("\n")


@db_app.command(name="search")
def db_search(
    query: str = typer.Argument(..., help="The search term to query."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The embedding model used to query context. If omitted, uses default_embedding config.",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="Maximum matches to return.",
    ),
):
    """Semantic search: queries the vector database for text segments matching the query."""
    if not auto_start_gateway():
        raise typer.Exit(1)

    chosen_model = model_name if model_name else DEFAULT_EMBEDDING
    if not chosen_model:
        models = get_local_models_info()
        emb_models = [m["name"] for m in models if "embedding" in m["name"].lower() or "bert" in m["name"].lower()]
        if emb_models:
            chosen_model = emb_models[0]

    if not chosen_model:
        console.print("[red]Error: No embedding model specified and no default embedding model configured.[/red]")
        raise typer.Exit(1)

    model_name = chosen_model

    # Pre-load the embedding model
    url_load = f"http://127.0.0.1:{HERD_PORT}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": model_name, "is_embedding": True}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load embedding model: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"Searching semantic index for '{query}'...")
    try:
        query_vector = asyncio.run(get_embedding(query, model_name))
        matches = search_vectors(query_vector, model_name, top_k=limit)
    except Exception as e:
        console.print(f"[red]Failed to perform semantic search: {e}[/red]")
        raise typer.Exit(1)

    if not matches:
        console.print("[yellow]No semantic matches found.[/yellow]")
        return

    table = Table(title=f"Semantic Search Matches for '{query}'")
    table.add_column("Similarity", style="green")
    table.add_column("Source File", style="cyan")
    table.add_column("Text Preview (Snippet)", style="white")

    for m in matches:
        filename = os.path.basename(m["file_path"])
        preview = m["text"][:80].replace("\n", " ") + "..."
        table.add_row(f"{m['similarity']:.3f}", filename, preview)

    console.print("\n")
    console.print(table)
    console.print("\n")


@db_app.command(name="remove")
def db_remove(
    path: str = typer.Argument(..., help="The file or directory path to remove from the index."),
):
    """Removes indexed chunks and files from the vector database."""
    from herd.services.rag import remove_indexed_path

    # Resolve absolute path to match DB entries
    abs_path = os.path.abspath(path)

    console.print(f"Removing indexed path [bold red]{abs_path}[/bold red] from database...")
    try:
        count = remove_indexed_path(abs_path)
        if count > 0:
            console.print(f"[bold green]Success![/bold green] Removed {count} chunks from the vector database.")
        else:
            console.print("[yellow]No indexed files found matching this path.[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to remove from database: {e}[/red]")
        raise typer.Exit(1)


app.add_typer(db_app)


def find_running_llm() -> Optional[str]:
    """Finds an active LLM (non-whisper, non-embedding) running in the gateway, uses default_llm config, or returns the first local model."""
    if is_gateway_running():
        try:
            res = httpx.get(f"http://127.0.0.1:{HERD_PORT}/v1/models/active", timeout=1.0)
            active = res.json()
            for m in active:
                if not m.get("is_whisper") and not m.get("is_embedding"):
                    return m["model"]
        except Exception:
            pass

    # Check configured default LLM
    if DEFAULT_LLM:
        return DEFAULT_LLM

    # Fallback to first downloaded LLM
    models = get_local_models_info()
    llms = [m["name"] for m in models if "whisper" not in m["name"].lower() and "mmproj" not in m["name"].lower()]
    if llms:
        return llms[0]
    return None


@app.command(name="copilot")
def copilot(
    instruction: str = typer.Argument(..., help="Natural language prompt describing what you want to execute in the terminal."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model identifier. If not specified, auto-detects the active or default LLM.",
    ),
):
    """Translates natural language into a shell command, explains it, and executes it on confirmation."""
    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No local LLM models found. Please pull a model first.[/red]")
        console.print("Example: [bold cyan]herd pull Qwen/Qwen3.5-0.8B:Q8_0[/bold cyan]")
        raise typer.Exit(1)

    # Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # Pre-load model
    url_load = f"http://127.0.0.1:{HERD_PORT}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_model}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    system_prompt = (
        "You are a terminal copilot. Translate the user's natural language request into a single executable shell command "
        "for the current operating system (Linux). "
        "Your response must be in JSON format with exactly two keys:\n"
        '1. "command": The single line shell command to execute.\n'
        '2. "explanation": A brief, one-sentence explanation of what the command does.\n\n'
        "Do not output any markdown formatting, backticks, or extra text. Output strictly valid JSON."
    )

    url_chat = f"http://127.0.0.1:{HERD_PORT}/v1/chat/completions"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ],
        "stream": False
    }

    console.print(f"Generating command using [bold cyan]{chosen_model}[/bold cyan]...")
    try:
        response = httpx.post(url_chat, json=payload, timeout=30.0)
        if response.status_code != 200:
            console.print(f"[red]Failed to generate command: {response.text}[/red]")
            raise typer.Exit(1)
        result = response.json()
        raw_text = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        console.print(f"[red]Error contacting Gateway: {e}[/red]")
        raise typer.Exit(1)

    # Clean markdown formatting if any
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        command = data["command"]
        explanation = data["explanation"]
    except Exception:
        console.print(f"[red]Error: Failed to parse generated response as JSON. Raw output was:[/red]\n{raw_text}")
        raise typer.Exit(1)

    # Print proposal
    from rich.panel import Panel
    from rich.console import Group
    panel_group = Group(
        f"[bold white]Command:[/bold white]\n  [bold green]{command}[/bold green]\n",
        f"[bold white]Explanation:[/bold white]\n  {explanation}"
    )
    console.print(Panel(
        panel_group,
        title="[bold cyan]Herd Shell Copilot Proposal[/bold cyan]",
        border_style="cyan",
        expand=False
    ))

    # Ask for confirmation
    confirm = typer.confirm("Do you want to execute this command?")
    if not confirm:
        console.print("[yellow]Aborted.[/yellow]")
        return

    console.print("\n[bold cyan]Running command...[/bold cyan]\n")
    try:
        subprocess.run(command, shell=True)
    except Exception as e:
        console.print(f"[red]Command execution failed: {e}[/red]")


@app.command(name="commit")
def commit(
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model identifier. If not specified, auto-detects the active or default LLM.",
    ),
):
    """Inspects Git repository modifications and generates conventional commits automatically offline."""
    # 1. Prerequisite checks
    if not shutil.which("git"):
        console.print("[red]Error: 'git' is not installed or not in PATH.[/red]")
        raise typer.Exit(1)
    if not os.path.exists(".git"):
        console.print("[red]Error: Current directory is not a Git repository.[/red]")
        raise typer.Exit(1)

    # 2. Get git diff
    diff_res = subprocess.run(["git", "diff"], capture_output=True, text=True)
    diff_text = diff_res.stdout.strip()

    # If no unstaged, check staged changes
    if not diff_text:
        diff_res = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True)
        diff_text = diff_res.stdout.strip()

    if not diff_text:
        console.print("[yellow]No changes detected in Git repository to commit.[/yellow]")
        return

    # Truncate diff if context limit exceeded
    if len(diff_text) > 10000:
        console.print("[yellow]Warning: Git diff is very large. Truncating to 10,000 characters.[/yellow]")
        diff_text = diff_text[:10000] + "\n\n... [TRUNCATED] ..."

    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No local LLM models found. Please pull a model first.[/red]")
        raise typer.Exit(1)

    # Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # Pre-load model
    url_load = f"http://127.0.0.1:{HERD_PORT}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_model}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    system_prompt = (
        "You are a Git commit message generator. Generate a clear, concise Conventional Commit message "
        "for the following git diff. "
        "The message must follow this format:\n"
        "<type>(<scope>): <short description>\n\n"
        "<body>\n\n"
        "Example:\n"
        "feat(RAG): add built-in local vector database search\n\n"
        "- Implement SQLite database for chunk vector indexing\n"
        "- Implement cosine similarity in pure Python\n\n"
        "Output only the commit message. Do not wrap in markdown or backticks."
    )

    url_chat = f"http://127.0.0.1:{HERD_PORT}/v1/chat/completions"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the diff:\n\n{diff_text}"}
        ],
        "stream": False
    }

    console.print(f"Generating commit message using [bold cyan]{chosen_model}[/bold cyan]...")
    try:
        response = httpx.post(url_chat, json=payload, timeout=30.0)
        if response.status_code != 200:
            console.print(f"[red]Failed to generate commit message: {response.text}[/red]")
            raise typer.Exit(1)
        result = response.json()
        commit_message = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        console.print(f"[red]Error contacting Gateway: {e}[/red]")
        raise typer.Exit(1)

    # Clean markdown wrapping if present
    if commit_message.startswith("```"):
        lines = commit_message.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        commit_message = "\n".join(lines).strip()

    from rich.panel import Panel
    console.print(Panel(
        commit_message,
        title="[bold cyan]Proposed Conventional Commit Message[/bold cyan]",
        border_style="cyan",
        expand=False
    ))

    confirm = typer.confirm("Would you like to commit these changes with this message?")
    if not confirm:
        console.print("[yellow]Aborted.[/yellow]")
        return

    # Write commit message to a temp file and run git commit
    import tempfile
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(commit_message)
        temp_path = f.name

    try:
        subprocess.run(["git", "commit", "-F", temp_path], check=True)
        console.print("[bold green]Success![/bold green] Changes committed.")
    except Exception as e:
        console.print(f"[red]Failed to execute commit: {e}[/red]")
    finally:
        os.remove(temp_path)


config_app = typer.Typer(name="config", help="Manage default models and environment configurations.")


@config_app.command(name="show")
def config_show():
    """Displays all current configurations, directory paths, and default models."""
    from herd.core.config import CONFIG_FILE, HERD_HOST, HERD_PORT, IDLE_TIMEOUT, LLAMA_SERVER_BIN, WHISPER_SERVER_BIN

    table = Table(title="Herd Configurations")
    table.add_column("Setting / Parameter", style="cyan")
    table.add_column("Value / Model Identifier", style="magenta")
    table.add_column("Source", style="green")

    # Load file values
    disk_config = load_config()

    def add_row(key: str, active_val, desc: str):
        source = "Environment / System"
        if key in disk_config:
            source = "config.json"
        elif active_val is None:
            source = "Not Configured"
        table.add_row(key, str(active_val) if active_val is not None else "-", source)

    add_row("default_llm", DEFAULT_LLM, "Default LLM model identifier")
    add_row("default_embedding", DEFAULT_EMBEDDING, "Default embedding model identifier")
    add_row("default_whisper", DEFAULT_WHISPER, "Default Whisper model identifier")
    add_row("HERD_HOST", HERD_HOST, "Gateway listen host")
    add_row("HERD_PORT", HERD_PORT, "Gateway listen port")
    add_row("HERD_IDLE_TIMEOUT", IDLE_TIMEOUT, "Gateway idle timeout in seconds")
    add_row("LLAMA_SERVER_BIN", LLAMA_SERVER_BIN, "Resolved llama-server path")
    add_row("WHISPER_SERVER_BIN", WHISPER_SERVER_BIN, "Resolved whisper-server path")
    add_row("Config File Path", CONFIG_FILE, "Location of config override JSON")

    console.print("\n")
    console.print(table)
    console.print("\nTo update defaults, run: [bold cyan]herd config set <key> <value>[/bold cyan]\n")


@config_app.command(name="set")
def config_set(
    key: str = typer.Argument(..., help="The setting key to modify (e.g. default_llm, default_embedding, default_whisper)."),
    value: str = typer.Argument(..., help="The value to assign to the key."),
):
    """Sets a configuration value in config.json."""
    valid_keys = {
        "default_llm",
        "default_embedding",
        "default_whisper",
        "HERD_PORT",
        "HERD_HOST",
        "HERD_IDLE_TIMEOUT",
        "LLAMA_SERVER_BIN",
        "WHISPER_SERVER_BIN"
    }

    # Normalize key to lower or match
    matched_key = None
    for k in valid_keys:
        if k.lower() == key.lower():
            matched_key = k
            break

    if not matched_key:
        console.print(f"[red]Error: Invalid config key '{key}'.[/red]")
        console.print(f"Supported keys: {', '.join(sorted(list(valid_keys)))}")
        raise typer.Exit(1)

    config = load_config()

    # Type conversion if necessary
    val_to_save = value
    if matched_key in ["HERD_PORT", "HERD_IDLE_TIMEOUT"]:
        try:
            val_to_save = int(value)
        except ValueError:
            console.print(f"[red]Error: Value for '{matched_key}' must be an integer.[/red]")
            raise typer.Exit(1)

    config[matched_key] = val_to_save
    try:
        save_config(config)
        console.print(f"[bold green]Success![/bold green] Configured [bold cyan]{matched_key}[/bold cyan] = [bold magenta]{val_to_save}[/bold magenta] in config.json.")
        console.print("[yellow]Please note: Restart running gateways or processes to apply port or timeout changes.[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to write configuration: {e}[/red]")
        raise typer.Exit(1)


app.add_typer(db_app)
app.add_typer(config_app)


def main():
    app()


if __name__ == "__main__":
    main()
