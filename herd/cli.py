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

from herd.core.config import HERD_PORT, HERD_LOGS_DIR, HERD_MODELS_DIR
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
    try:
        response = httpx.get(f"http://127.0.0.1:{HERD_PORT}/health", timeout=1.0)
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
    port: int = typer.Option(
        HERD_PORT, "--port", "-p", help="Port to run the gateway server on."
    ),
):
    """Starts the central Herd API Gateway server."""
    # Ensure gateway port is set in env so server.py knows about it
    os.environ["HERD_PORT"] = str(port)
    console.print(
        f"[bold green]Starting Herd API Gateway on port {port}...[/bold green]"
    )
    # Correct path to the FastAPI app module under the new package layout
    uvicorn.run("herd.api.server:app", host="127.0.0.1", port=port, log_level="info")


def main():
    app()


if __name__ == "__main__":
    main()
