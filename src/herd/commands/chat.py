import asyncio
import json
import os
import time
from typing import Optional

import httpx
import typer
from rich.table import Table

from herd.core.config import (
    HERD_HOST,
    HERD_PORT,
)
from herd.core.utils import (
    auto_start_gateway,
    console,
    find_running_llm,
    get_gateway_url,
    pull_model_async,
)
from herd.services.downloader import resolve_model_path
from herd.services.rag import get_embedding, search_vectors

DEFAULT_BENCHMARK_PROMPTS = [
    "Write a quick Python script to download a URL to a file.",
    "Explain what semantic RAG search is in 2 sentences.",
    "How does a local reverse proxy help connect multiple machine configurations?",
]


async def stream_chat_completions(model_name: str, messages: list) -> str:
    """Sends a streaming chat completion request to Herd gateway."""
    url = f"{get_gateway_url()}/v1/chat/completions"
    payload = {"model": model_name, "messages": messages, "stream": True}

    assistant_response = ""
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                err_body = await response.aread()
                console.print(
                    f"\n[red]API request failed ({response.status_code}): {err_body.decode()}[/red]"
                )
                raise typer.Exit(1)

            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                # Split chunks into SSE lines
                lines = chunk.decode("utf-8", errors="ignore").split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
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
    console.print(f"\n[bold green]Chatting with {model_name} (Herd Gateway)[/bold green]")
    console.print(
        "Type [bold cyan]/help[/bold cyan] to see available commands. Press Ctrl+C to stop generation.\n"
    )

    # Load embedding model if RAG context is active
    if context_model:
        url_load = f"{get_gateway_url()}/v1/models/load"
        try:
            httpx.post(
                url_load,
                json={"model": context_model, "is_embedding": True},
                timeout=45.0,
            )
            console.print(
                f"[dim]RAG Active: Retrieving context from embedding model '{context_model}'[/dim]\n"
            )
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
                    console.print(
                        "  [bold white]/help[/bold white]               - Show this help menu."
                    )
                    console.print(
                        "  [bold white]/clear[/bold white] or [bold white]/reset[/bold white]   - Clear the chat history."
                    )
                    console.print(
                        "  [bold white]/system <prompt>[/bold white]     - Set or update the system prompt."
                    )
                    console.print(
                        "  [bold white]/export [filename][/bold white]   - Export the chat history to a Markdown file."
                    )
                    console.print(
                        "  [bold white]/exit[/bold white] or [bold white]/quit[/bold white]     - Exit the chat session.\n"
                    )
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
                    console.print(
                        f"[yellow]System prompt updated to:[/yellow] [italic]{arg}[/italic]"
                    )
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
                        retrieved_context = "\n\n".join(
                            [
                                f"Source: {os.path.basename(m['file_path'])}\n{m['text']}"
                                for m in matches
                            ]
                        )
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
                        ),
                    }
                assistant_response = await stream_chat_completions(model_name, payload_messages)
                messages.append({"role": "assistant", "content": assistant_response})
            except KeyboardInterrupt:
                print("\n[yellow]Generation interrupted.[/yellow]")
                messages.append({"role": "assistant", "content": "[Generation Interrupted]"})
            except Exception as e:
                console.print(f"\n[red]Error during generation: {e}[/red]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting chat session...[/yellow]")
            break


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
        help="Run as a BERT embeddings vector model server.",
    ),
    idle_timeout: int = typer.Option(
        300,
        "--idle-timeout",
        help="Model idle timeout in seconds (stops model process when idle).",
    ),
    context_model: Optional[str] = typer.Option(
        None,
        "--context",
        "-c",
        help="Pre-load and query context semantically from this embedding model on every turn.",
    ),
):
    """Starts a model process in the background and launches a chat REPL session (LLMs only)."""
    # 1. Resolve model defaults
    chosen_model = model_name
    if not chosen_model:
        chosen_model = find_running_llm()

    if not chosen_model:
        console.print(
            "[red]Error: No model name specified and no suitable default model configured.[/red]"
        )
        console.print(
            "Please pull a model first or configure defaults using [bold cyan]herd config set[/bold cyan]."
        )
        raise typer.Exit(1)

    model_name = chosen_model

    # 2. Check if model exists locally. If not, prompt to download (only for local gateway, skipping cloud providers)
    from herd.core.config import settings

    is_cloud = False
    if ":" in model_name:
        parts = model_name.split(":", 1)
        if parts[0] in settings.providers:
            is_cloud = True

    if not is_cloud:
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
    url = f"{get_gateway_url()}/v1/models/load"
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
        port = data.get("port", "proxy")
    except Exception as e:
        console.print(f"[red]Error loading model: {e}[/red]")
        raise typer.Exit(1)

    # 4. Enter chat REPL or display server status
    if whisper or "whisper" in model_name.lower():
        console.print("\n[bold green]Whisper model loaded successfully![/bold green]")
        console.print(f"Whisper server running internally on port [bold cyan]{port}[/bold cyan].")
        console.print("You can send transcription requests to the Gateway:")
        console.print(
            f"  [bold white]POST http://127.0.0.1:{HERD_PORT}/v1/audio/transcriptions[/bold white]"
        )
        console.print(
            f"  [bold white]POST http://127.0.0.1:{HERD_PORT}/v1/audio/translations[/bold white]"
        )
    elif embedding or "embedding" in model_name.lower() or "bert" in model_name.lower():
        console.print("\n[bold green]Embedding model loaded successfully![/bold green]")
        console.print(f"Model server running internally on port [bold cyan]{port}[/bold cyan].")
        console.print("You can send embedding requests to the Gateway:")
        console.print(f"  [bold white]POST http://127.0.0.1:{HERD_PORT}/v1/embeddings[/bold white]")
    else:
        asyncio.run(chat_interactive(model_name, context_model))


async def run_benchmark_async(model_name: str, custom_prompts: Optional[list[str]], rounds: int):
    """Runs the benchmark suite async."""
    # 1. Prerequisite checks
    if not auto_start_gateway():
        raise typer.Exit(1)

    # 2. Check if model exists locally (only for local gateway, skipping cloud providers)
    from herd.core.config import settings

    is_cloud = False
    if ":" in model_name:
        parts = model_name.split(":", 1)
        if parts[0] in settings.providers:
            is_cloud = True

    if not is_cloud:
        try:
            resolve_model_path(model_name)
        except FileNotFoundError:
            console.print(
                f"[red]Error: Model '{model_name}' not found locally. Please pull it first.[/red]"
            )
            raise typer.Exit(1)

    # 3. Load model in gateway
    console.print(
        f"Loading [bold cyan]{model_name}[/bold cyan] and running benchmark suite ({rounds} rounds per prompt)..."
    )
    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": model_name}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to pre-load model: {e}[/red]")
        raise typer.Exit(1)

    prompts = custom_prompts if custom_prompts else DEFAULT_BENCHMARK_PROMPTS
    results = []

    for idx, prompt in enumerate(prompts):
        console.print(
            f'\n[bold magenta]Prompt {idx + 1}/{len(prompts)}:[/bold magenta] [italic]"{prompt[:60]}..."[/italic]'
        )

        ttfts = []
        speeds = []
        mems = []
        cpus = []

        for r in range(rounds):
            print(f"  Round {r + 1}/{rounds}...", end="", flush=True)

            url_chat = f"{get_gateway_url()}/v1/chat/completions"
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
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
                    res = httpx.get(f"http://{host}:{HERD_PORT}/v1/models/active", timeout=1.0)
                    active = res.json()
                    for m in active:
                        if m["model"] == model_name:
                            cpus.append(m.get("cpu_percent", 0.0))
                            # Convert MB/GB to float GB
                            mem_str = m.get("memory_str", "0 B")
                            if "GB" in mem_str:
                                mems.append(float(mem_str.replace("GB", "").strip()))
                            elif "MB" in mem_str:
                                mems.append(float(mem_str.replace("MB", "").strip()) / 1024.0)
                except Exception:
                    pass

            stats_task = asyncio.create_task(get_stats_during_inference())

            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", url_chat, json=payload) as response:
                        if response.status_code != 200:
                            print("[red]failed[/red]")
                            continue
                        async for chunk in response.aiter_bytes():
                            if not first_token_time:
                                first_token_time = time.time()
                            # Parse token counts
                            lines = chunk.decode("utf-8", errors="ignore").split("\n")
                            for line in lines:
                                if line.startswith("data: ") and "[DONE]" not in line:
                                    token_count += 1
                await stats_task
            except Exception as e:
                print(f"[red]error: {e}[/red]")
                continue

            end_time = time.time()
            total_duration = end_time - start_time

            # TTFT (Time to First Token)
            ttft = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
            ttfts.append(ttft)

            # Speed (Tokens per second)
            generation_time = end_time - first_token_time if first_token_time else total_duration
            speed = token_count / generation_time if generation_time > 0 else 0.0
            speeds.append(speed)

            print(" done.")

        # Calculate averages for this prompt
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        avg_cpu = sum(cpus) / len(cpus) if cpus else 0.0
        avg_mem = sum(mems) / len(mems) if mems else 0.0

        results.append(
            {
                "prompt": prompt,
                "ttft": f"{avg_ttft:.1f} ms",
                "speed": f"{avg_speed:.1f} t/s",
                "memory": f"{avg_mem:.2f} GB" if avg_mem > 0 else "-",
                "cpu": f"{avg_cpu:.1f}%" if avg_cpu > 0 else "-",
            }
        )

    # Render summary table
    table = Table(title=f"Benchmark Summary: {model_name}")
    table.add_column("Prompt / Query", style="cyan")
    table.add_column("Avg TTFT", style="yellow")
    table.add_column("Avg Speed", style="green")
    table.add_column("Avg Memory (RAM/VRAM)", style="magenta")
    table.add_column("Avg CPU %", style="blue")

    for r in results:
        table.add_row(r["prompt"][:50] + "...", r["ttft"], r["speed"], r["memory"], r["cpu"])

    console.print("\n")
    console.print(table)


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
