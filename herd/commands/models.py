import os
import time
import asyncio
import subprocess
import shutil
import httpx
import typer
from typing import Optional
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.console import Group

from herd.core.config import (
    HERD_HOST,
    HERD_PORT,
    HERD_LOGS_DIR,
)
from herd.core.utils import (
    console,
    get_gateway_url,
    is_gateway_running,
    get_local_models_info,
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


def stop(
    model_name: Optional[str] = typer.Argument(
        None, help="Model identifier to stop. Required unless --all is specified."
    ),
    stop_all: bool = typer.Option(
        False, "--all", "-a", help="Stop all running model processes."
    ),
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
                console.print(
                    f"[red]Failed to query active models: {active_res.text}[/red]"
                )
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
                    console.print(
                        f"[green]Successfully stopped model '{m_name}'.[/green]"
                    )
                else:
                    console.print(
                        f"[red]Failed to stop model '{m_name}': {response.text}[/red]"
                    )
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
                console.print(
                    f"[green]Successfully stopped model '{model_name}'.[/green]"
                )
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
            "Speech"
            if a.get("is_whisper")
            else ("Embedding" if a.get("is_embedding") else "LLM")
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
        console.print(
            "[yellow]No stats collected yet. Send some requests first![/yellow]"
        )
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
        console.print(
            "[yellow]Log directory does not exist. Nothing to clean.[/yellow]"
        )
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
        console.print(
            "[green]No inactive logs found. Your log directory is clean![/green]"
        )
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

    console.print(
        f"[green]Successfully deleted {deleted_count} inactive log file(s).[/green]"
    )


def search(
    query: str = typer.Argument(
        ..., help="Search term for GGUF models on Hugging Face Hub."
    ),
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


def find_llama_quantize():
    """Finds the compiled llama-quantize binary."""
    # Check ~/.herd/src/llama.cpp/build/bin/llama-quantize
    local_path = os.path.expanduser("~/.herd/src/llama.cpp/build/bin/llama-quantize")
    if os.path.exists(local_path):
        return local_path
    # Fallback to PATH
    return shutil.which("llama-quantize")


def quantize(
    input_file: str = typer.Argument(
        ..., help="Path to the source GGUF file (e.g. FP16/FP32)."
    ),
    output_file: str = typer.Argument(
        ..., help="Path to save the output quantized GGUF file."
    ),
    method: str = typer.Argument(
        ..., help="Quantization method (e.g. Q4_K_M, Q8_0, Q5_K_M)."
    ),
):
    """Quantizes (compresses) a GGUF model file locally using the compiled llama-quantize binary."""
    if not os.path.exists(input_file):
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    quant_bin = find_llama_quantize()
    if not quant_bin:
        console.print("[red]Error: 'llama-quantize' binary not found.[/red]")
        console.print(
            "Please make sure you have run [bold cyan]herd setup[/bold cyan] to build the compilation tools locally."
        )
        raise typer.Exit(1)

    console.print(
        f"Quantizing [bold cyan]{input_file}[/bold cyan] to [bold green]{output_file}[/bold green] using method [bold yellow]{method}[/bold yellow]..."
    )
    try:
        cmd = [quant_bin, input_file, output_file, method]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()

        if process.returncode == 0:
            console.print(
                f"\n[bold green]Success![/bold green] Quantized model saved to {output_file}"
            )
        else:
            console.print(
                f"\n[red]Quantization failed with exit code: {process.returncode}[/red]"
            )
            raise typer.Exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Quantization interrupted.[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error running quantization: {e}[/red]")
        raise typer.Exit(1)


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
            return Panel(
                "[yellow]Herd Gateway is not running.[/yellow]", title="Herd Top"
            )

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
