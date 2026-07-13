import os
import sys
import time
import subprocess
import httpx
import typer
from typing import Optional
from rich.console import Console

from herd.core.config import (
    HERD_HOST,
    HERD_PORT,
    HERD_LOGS_DIR,
    HERD_MODELS_DIR,
    DEFAULT_LLM,
)
from herd.services.downloader import (
    list_hf_repository_files,
    download_file,
    parse_model_identifier,
)

console = Console()

# Shared HTTP clients for connection pooling
_shared_client = None
_shared_async_client = None


def get_http_client() -> httpx.Client:
    """Returns a shared, pooled synchronous HTTP client."""
    global _shared_client
    if _shared_client is None:
        _shared_client = httpx.Client(timeout=10.0)
    return _shared_client


def get_async_http_client() -> httpx.AsyncClient:
    """Returns a shared, pooled asynchronous HTTP client."""
    global _shared_async_client
    if _shared_async_client is None:
        _shared_async_client = httpx.AsyncClient(timeout=None)
    return _shared_async_client


def close_http_clients():
    """Closes and cleans up shared HTTP clients."""
    global _shared_client, _shared_async_client
    if _shared_client is not None:
        try:
            _shared_client.close()
        except Exception:
            pass
        _shared_client = None
    if _shared_async_client is not None:
        import asyncio

        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_shared_async_client.aclose())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(_shared_async_client.aclose())
        except Exception:
            pass
        _shared_async_client = None


def get_gateway_url() -> str:
    """Returns the API Gateway base URL."""
    return f"http://127.0.0.1:{HERD_PORT}"


def is_gateway_running() -> bool:
    """Checks if the Herd gateway server is currently running."""
    client = get_http_client()

    host = HERD_HOST
    if host == "0.0.0.0":
        host = "127.0.0.1"
    try:
        response = client.get(f"http://{host}:{HERD_PORT}/health", timeout=1.0)
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


def find_running_llm() -> Optional[str]:
    """Finds the default configured LLM, an active running LLM on the gateway, or the first local model."""
    # 1. Check configured default LLM
    if DEFAULT_LLM:
        return DEFAULT_LLM

    # 2. Check active running models in the gateway
    if is_gateway_running():
        try:
            res = httpx.get(f"{get_gateway_url()}/v1/models/active", timeout=1.0)
            active = res.json()
            for m in active:
                if not m.get("is_whisper") and not m.get("is_embedding"):
                    return m["model"]
        except Exception:
            pass

    # 3. Fallback to first downloaded LLM
    models = get_local_models_info()
    llms = [
        m["name"]
        for m in models
        if "whisper" not in m["name"].lower() and "mmproj" not in m["name"].lower()
    ]
    if llms:
        return llms[0]
    return None


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
            console.print(f"[{idx}] {f}")

        try:
            choice = typer.prompt("\nEnter choice index", type=int)
            if choice < 0 or choice >= len(model_files):
                console.print("[red]Invalid choice.[/red]")
                raise typer.Exit(1)
            chosen_file = model_files[choice]
        except Exception:
            console.print("[red]Invalid input.[/red]")
            raise typer.Exit(1)

    # Build download URL and destination path
    download_url = f"https://huggingface.co/{author}/{repo}/resolve/main/{chosen_file}"
    dest_path = os.path.join(HERD_MODELS_DIR, "huggingface", author, repo, chosen_file)

    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        console.print(
            f"[yellow]Model file already exists locally at: {dest_path}. Skipping download.[/yellow]"
        )
        return

    console.print(f"Downloading file: [bold cyan]{chosen_file}[/bold cyan]")
    try:
        await download_file(download_url, dest_path, chosen_file)
        console.print(
            f"[green]Successfully downloaded and saved model to: {dest_path}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Failed to download file: {e}[/red]")
        raise typer.Exit(1)
