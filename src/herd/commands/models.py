import asyncio
import json
import os
import shutil
import time
from typing import Optional

import httpx
import typer
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from herd.core.config import (
    HERD_HOST,
    HERD_LOGS_DIR,
    HERD_PORT,
)
from herd.core.utils import (
    console,
    get_gateway_url,
    get_local_models_info,
    is_gateway_running,
    pull_model_async,
)


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
        models = [m for m in models if m.get("provider", "").lower() == provider.lower()]

    if filter_query:
        models = [
            m
            for m in models
            if filter_query.lower() in m["name"].lower()
            or filter_query.lower() in m["filename"].lower()
        ]

    if not models:
        if filter_query or provider:
            console.print("[yellow]No models matched the specified filter criteria.[/yellow]")
        else:
            console.print(
                "[yellow]No models found. Use 'herd pull <model_name>' to download some.[/yellow]"
            )
        return

    table = Table(title="Downloaded Models")
    table.add_column("Provider", style="cyan")
    table.add_column("Model Name / Identifier", style="magenta")
    table.add_column("File Name", style="green")
    table.add_column("File Size", style="yellow")

    for m in models:
        table.add_row(m.get("provider", "local"), m["name"], m["filename"], m["size"])

    console.print(table)


def pull(
    model_name: str = typer.Argument(
        ...,
        help="Model identifier format 'author/repo[:tag]' (e.g. unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M)",
    ),
):
    """Downloads a GGUF or BIN model from Hugging Face."""
    asyncio.run(pull_model_async(model_name))


def rm(
    model_name: str = typer.Argument(
        ...,
        help="Model identifier to delete (e.g. 'unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M' or 'unsloth/Qwen3.5-0.8B-GGUF')",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Deletes a downloaded model or repository from local storage."""
    from herd.core.config import HERD_MODELS_DIR
    from herd.services.downloader import parse_model_identifier

    try:
        author, repo, tag = parse_model_identifier(model_name)
        repo_dir = os.path.join(HERD_MODELS_DIR, "huggingface", author, repo)
    except Exception:
        # Check if it's a local path inside HERD_MODELS_DIR
        abs_path = os.path.abspath(model_name)
        if not abs_path.startswith(os.path.abspath(HERD_MODELS_DIR)):
            console.print(
                f"[red]Error: Cannot delete '{model_name}'. 'herd rm' only deletes models stored inside {HERD_MODELS_DIR}.[/red]"
            )
            raise typer.Exit(1)

        if not os.path.exists(abs_path):
            console.print(f"[red]Error: Model path '{model_name}' does not exist.[/red]")
            raise typer.Exit(1)

        if not yes:
            confirm = typer.confirm(
                f"Are you sure you want to delete local model path '{model_name}'?"
            )
            if not confirm:
                console.print("[yellow]Deletion cancelled.[/yellow]")
                return

        try:
            if os.path.isdir(abs_path):
                shutil.rmtree(abs_path)
            else:
                os.remove(abs_path)
            console.print(f"[green]Successfully deleted local model path '{model_name}'.[/green]")
            return
        except Exception as e:
            console.print(f"[red]Error deleting '{model_name}': {e}[/red]")
            raise typer.Exit(1)

    if not os.path.exists(repo_dir):
        console.print(f"[yellow]Model repository directory does not exist: {repo_dir}[/yellow]")
        return

    if tag:
        files = [f for f in os.listdir(repo_dir) if os.path.isfile(os.path.join(repo_dir, f))]
        model_files = [f for f in files if f.endswith(".gguf") or f.endswith(".bin")]

        tagged_files = [
            f for f in model_files if tag.lower() in f.lower() and "mmproj" not in f.lower()
        ]
        if not tagged_files:
            tagged_files = [f for f in model_files if tag.lower() in f.lower()]

        if not tagged_files:
            console.print(f"[yellow]No files matching tag '{tag}' found in {repo_dir}.[/yellow]")
            return

        target_file = os.path.join(repo_dir, tagged_files[0])
        if not yes:
            confirm = typer.confirm(
                f"Are you sure you want to delete model file '{tagged_files[0]}'?"
            )
            if not confirm:
                console.print("[yellow]Deletion cancelled.[/yellow]")
                return

        try:
            os.remove(target_file)
            console.print(f"[green]Successfully deleted model file '{tagged_files[0]}'.[/green]")

            # If the directory is now empty, clean it up
            remaining = os.listdir(repo_dir)
            if not remaining:
                shutil.rmtree(repo_dir)
                author_dir = os.path.dirname(repo_dir)
                if os.path.exists(author_dir) and not os.listdir(author_dir):
                    os.rmdir(author_dir)
        except Exception as e:
            console.print(f"[red]Error deleting model file: {e}[/red]")
            raise typer.Exit(1)
    else:
        if not yes:
            confirm = typer.confirm(
                f"Are you sure you want to delete the entire model repository '{model_name}'?"
            )
            if not confirm:
                console.print("[yellow]Deletion cancelled.[/yellow]")
                return

        try:
            shutil.rmtree(repo_dir)
            console.print(f"[green]Successfully deleted model repository '{model_name}'.[/green]")

            author_dir = os.path.dirname(repo_dir)
            if os.path.exists(author_dir) and not os.listdir(author_dir):
                os.rmdir(author_dir)
        except Exception as e:
            console.print(f"[red]Error deleting model repository: {e}[/red]")
            raise typer.Exit(1)


def stop(
    model_name: Optional[str] = typer.Argument(
        None, help="Model identifier to stop. Required unless --all is specified."
    ),
    stop_all: bool = typer.Option(False, "--all", "-a", help="Stop all running model processes."),
):
    """Stops a running model process."""
    if not is_gateway_running():
        console.print("[yellow]Herd API gateway is not running.[/yellow]")
        return

    url = f"{get_gateway_url()}/v1/models/unload"

    if stop_all:
        # Fetch active models
        active_url = f"{get_gateway_url()}/v1/models/active"
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
            console.print(
                "[red]Error: Please specify a model name, or use --all (-a) to stop all running models.[/red]"
            )
            raise typer.Exit(1)

        try:
            response = httpx.post(url, json={"model": model_name})
            if response.status_code == 200:
                console.print(f"[green]Successfully stopped model '{model_name}'.[/green]")
            else:
                console.print(f"[red]Failed to stop model: {response.text}[/red]")
        except Exception as e:
            console.print(f"[red]Error stopping model: {e}[/red]")


def ps():
    """Lists currently active running model processes in the gateway."""
    if not is_gateway_running():
        console.print("[yellow]Herd API gateway is not running.[/yellow]")
        return

    url = f"{get_gateway_url()}/v1/models/active"
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
    table.add_column("Memory", style="blue")
    table.add_column("Idle Time", style="white")

    for a in active:
        m_type = (
            "Speech" if a.get("is_whisper") else ("Embedding" if a.get("is_embedding") else "LLM")
        )
        idle_str = f"{a['idle_seconds']}s"
        cpu_str = f"{a.get('cpu_percent', 0.0)}%"
        mem_str = a.get("memory_str", "0 MB")
        table.add_row(a["model"], str(a["port"]), m_type, cpu_str, mem_str, idle_str)

    console.print(table)


def show_stats():
    """Displays cumulative request, token, and performance stats for all models."""
    if not is_gateway_running():
        console.print("[yellow]Herd API gateway is not running.[/yellow]")
        return

    url = f"{get_gateway_url()}/v1/models/stats"
    try:
        response = httpx.get(url)
        response.raise_for_status()
        stats = response.json()
    except Exception as e:
        console.print(f"[red]Failed to query model stats: {e}[/red]")
        return

    if not stats:
        console.print("[yellow]No stats collected yet. Send some requests first![/yellow]")
        return

    table = Table(title="Herd Model Usage Statistics")
    table.add_column("Model", style="cyan")
    table.add_column("Requests", style="green")
    table.add_column("Prompt Tokens", style="blue")
    table.add_column("Completion Tokens", style="magenta")
    table.add_column("Avg Latency", style="yellow")
    table.add_column("Avg Speed", style="white")

    for model, data in stats.items():
        req_str = f"{data['request_count']}"
        if data["error_count"] > 0:
            req_str += f" ({data['error_count']} err)"

        lat = data.get("avg_latency_ms", 0.0)
        lat_str = f"{lat / 1000.0:.2f}s" if lat > 0 else "-"

        speed = data.get("avg_speed_tps", 0.0)
        speed_str = f"{speed:.1f} t/s" if speed > 0 else "-"

        table.add_row(
            model,
            req_str,
            f"{data['prompt_tokens']:,}",
            f"{data['completion_tokens']:,}",
            lat_str,
            speed_str,
        )

    console.print(table)


def history_cmd(
    request_id: Optional[int] = typer.Argument(
        None, help="Optional Request ID to inspect full sent & received message transcript."
    ),
    limit: int = typer.Option(
        20, "--limit", "-n", help="Number of recent request history entries to display."
    ),
):
    """Displays recent request history logs or inspects full messages for a specific request ID."""
    if not is_gateway_running():
        console.print("[yellow]Herd API gateway is not running.[/yellow]")
        return

    # Single Request ID Detail Inspection
    if request_id is not None:
        url = f"{get_gateway_url()}/v1/models/history/{request_id}"
        try:
            response = httpx.get(url)
            if response.status_code == 404:
                console.print(f"[red]Request history record #{request_id} not found.[/red]")
                return
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            console.print(f"[red]Failed to query request detail: {e}[/red]")
            return

        status_badge = (
            "[bold red]ERROR[/bold red]" if data.get("is_error") else "[bold green]OK[/bold green]"
        )
        console.print(
            f"\n🔍 [bold cyan]Request Inspection #{data.get('id')}[/bold cyan] ({status_badge})"
        )
        console.print(f"  [bold white]Timestamp:[/bold white] {data.get('timestamp')}")
        console.print(f"  [bold white]Model:[/bold white]     {data.get('model_name')}")
        console.print(f"  [bold white]Endpoint:[/bold white]  {data.get('endpoint')}")
        console.print(
            f"  [bold white]Tokens:[/bold white]    Prompt: {data.get('prompt_tokens')} | Gen: {data.get('completion_tokens')}"
        )
        console.print(f"  [bold white]Duration:[/bold white]  {data.get('duration_sec')}s\n")

        # Sent Prompt Messages
        prompt_content = data.get("full_prompt")
        if isinstance(prompt_content, (dict, list)):
            prompt_str = json.dumps(prompt_content, indent=2)
        else:
            prompt_str = str(prompt_content or "")

        console.print(
            Panel(prompt_str, title="📤 Sent Payload (Messages / Prompt)", border_style="cyan")
        )

        # Received Response Content
        resp_content = data.get("full_response")
        if isinstance(resp_content, (dict, list)):
            resp_str = json.dumps(resp_content, indent=2)
        else:
            resp_str = str(resp_content or "")

        console.print(
            Panel(resp_str, title="📥 Received Response (Completion)", border_style="magenta")
        )
        return

    # List Summary Table
    url = f"{get_gateway_url()}/v1/models/history?limit={limit}"
    try:
        response = httpx.get(url)
        response.raise_for_status()
        history = response.json()
    except Exception as e:
        console.print(f"[red]Failed to query request history: {e}[/red]")
        return

    if not history:
        console.print("[yellow]No history recorded yet. Send some LLM requests first![/yellow]")
        return

    table = Table(title=f"Herd Recent Request History (Last {len(history)})")
    table.add_column("ID", style="bold cyan")
    table.add_column("Timestamp", style="dim white")
    table.add_column("Model", style="cyan")
    table.add_column("Endpoint", style="magenta")
    table.add_column("Prompt Snippet", style="white")
    table.add_column("Tokens (P / C)", style="blue")
    table.add_column("Duration", style="yellow")
    table.add_column("Status", style="bold")

    for item in history:
        status_str = (
            "[bold red]ERROR[/bold red]" if item.get("is_error") else "[bold green]OK[/bold green]"
        )
        tok_str = f"{item.get('prompt_tokens', 0)} / {item.get('completion_tokens', 0)}"
        dur_str = f"{item.get('duration_sec', 0.0):.2f}s"
        snippet = item.get("prompt_snippet", "")
        if len(snippet) > 30:
            snippet = snippet[:27] + "..."
        if not snippet:
            snippet = "-"

        table.add_row(
            str(item.get("id", "-")),
            item.get("timestamp", "-"),
            item.get("model_name", "-"),
            item.get("endpoint", "-"),
            snippet,
            tok_str,
            dur_str,
            status_str,
        )

    console.print(table)
    console.print("[dim]Run 'herd history <id>' to inspect full sent & received messages.[/dim]\n")


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
    console.print(
        f"Searching Hugging Face Hub for GGUF models matching '[bold cyan]{query}[/bold cyan]'..."
    )
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
    console.print(
        "\nTo pull a model, use: [bold cyan]herd pull <repository_id>:<tag>[/bold cyan]\n"
    )


def top():
    """Opens a real-time monitor displaying active model servers and resource utilization."""
    host = HERD_HOST
    if host == "0.0.0.0":
        host = "127.0.0.1"
    url = f"http://{host}:{HERD_PORT}/v1/models/active"

    def make_display() -> Panel:
        try:
            res = httpx.get(url, timeout=1.0)
            if res.status_code != 200:
                return Panel(
                    "[red]Error: Gateway returned unsuccessful status.[/red]",
                    title="Herd Top",
                )
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
        total_mem_str = (
            f"{total_mem_gb:.2f} GB"
            if total_mem_gb >= 0.1
            else f"{total_mem / (1024 * 1024):.1f} MB"
        )

        summary = (
            f"Active Models: [bold white]{len(active)}[/bold white] | "
            f"Total CPU: [bold white]{total_cpu:.1f}%[/bold white] | "
            f"Total Memory: [bold white]{total_mem_str}[/bold white]"
        )

        return Panel(
            Group(summary, "", table),
            title="[bold green]Herd Top — Real-Time Model Monitor[/bold green]",
            border_style="green",
            subtitle="[dim]Press Ctrl+C to exit[/dim]",
        )

    try:
        with Live(make_display(), refresh_per_second=2) as live:
            while True:
                time.sleep(0.5)
                live.update(make_display())
    except KeyboardInterrupt:
        console.print("\n[yellow]Exiting monitor...[/yellow]")
