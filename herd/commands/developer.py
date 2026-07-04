import os
import json
import subprocess
import shutil
import httpx
import typer
from typing import Optional
from rich.panel import Panel
from rich.console import Group

from herd.core.utils import (
    console,
    get_gateway_url,
    auto_start_gateway,
    find_running_llm,
)


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


def suggest():
    """Analyzes system hardware (RAM and VRAM) and suggests compatible LLMs and Whisper models."""
    import psutil

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

    # Whisper STT recommendations
    if ram_gb >= 16.0 or vram_gb >= 8.0:
        whisper_rec = "ggerganov/whisper.cpp:ggml-large-v3-turbo.bin"
        whisper_desc = "Large multilingual STT model. Highest accuracy."
        whisper_multilingual = "ggerganov/whisper.cpp:ggml-large-v3-turbo.bin"
        whisper_multi_desc = "Highly accurate multilingual support."
    elif ram_gb >= 8.0 or vram_gb >= 4.0:
        whisper_rec = "ggerganov/whisper.cpp:ggml-base.en.bin"
        whisper_desc = "English-only base model. Fast and balanced accuracy."
        whisper_multilingual = "ggerganov/whisper.cpp:ggml-base.bin"
        whisper_multi_desc = "Good speed/accuracy multilingual model."
    else:
        whisper_rec = "ggerganov/whisper.cpp:ggml-tiny.en.bin"
        whisper_desc = "Tiny english model. Fast but lower accuracy."
        whisper_multilingual = "ggerganov/whisper.cpp:ggml-tiny.bin"
        whisper_multi_desc = "Fastest multilingual model."

    report = []
    report.append(f"System RAM: [bold white]{ram_gb:.1f} GB[/bold white]")
    if gpu:
        report.append(f"GPU Detected: [bold white]{gpu['name']}[/bold white] | VRAM: [bold white]{vram_gb:.1f} GB[/bold white] (Driver: {gpu['driver']})")
    else:
        report.append("GPU Detected: [bold white]None / Integrated[/bold white]")

    report.append("")
    report.append("[bold green]🤖 Recommended Chat LLM (CPU-only):[/bold green]")
    report.append(f"  Model: [white]{cpu_llm}[/white] ({cpu_desc})")
    report.append(f"  Pull Command: [bold cyan]herd pull {cpu_llm}[/bold cyan]")
    report.append("")

    if gpu_llm:
        report.append("[bold green]⚡ Recommended Chat LLM (GPU-accelerated):[/bold green]")
        report.append(f"  Model: [white]{gpu_llm}[/white] ({gpu_desc})")
        report.append(f"  Pull Command: [bold cyan]herd pull {gpu_llm}[/bold cyan]")
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
    url_load = f"{get_gateway_url()}/v1/models/load"
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

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
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
    url_load = f"{get_gateway_url()}/v1/models/load"
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

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
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


def review(
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model identifier. If omitted, uses active or default LLM.",
    ),
    pre_commit: bool = typer.Option(
        False,
        "--pre-commit",
        help="Run in pre-commit hook mode (exit code 1 if critical issues are found, stages-only diff).",
    ),
    install_hook: bool = typer.Option(
        False,
        "--install",
        "-i",
        help="Install herd review as a local Git pre-commit hook.",
    ),
):
    """Inspects Git repository modifications and audits code changes for quality, bugs, and security risks."""
    # 1. Install hook option
    if install_hook:
        if not os.path.exists(".git"):
            console.print("[red]Error: Current directory is not a Git repository.[/red]")
            raise typer.Exit(1)
        hook_dir = ".git/hooks"
        os.makedirs(hook_dir, exist_ok=True)
        hook_path = os.path.join(hook_dir, "pre-commit")

        script = "#!/bin/sh\nherd review --pre-commit\n"
        try:
            with open(hook_path, "w") as f:
                f.write(script)
            os.chmod(hook_path, 0o755)
            console.print("[bold green]Success![/bold green] Installed pre-commit hook at .git/hooks/pre-commit")
            return
        except Exception as e:
            console.print(f"[red]Failed to install pre-commit hook: {e}[/red]")
            raise typer.Exit(1)

    # 2. Prerequisite checks
    if not shutil.which("git"):
        console.print("[red]Error: 'git' is not installed or not in PATH.[/red]")
        raise typer.Exit(1)
    if not os.path.exists(".git"):
        console.print("[red]Error: Current directory is not a Git repository.[/red]")
        raise typer.Exit(1)

    # 3. Get git diff
    if pre_commit:
        # Pre-commit hook only audits staged changes
        diff_res = subprocess.run(["git", "diff", "--staged"], capture_output=True, text=True)
    else:
        # Normal mode audits both staged + unstaged changes
        diff_res = subprocess.run(["git", "diff", "HEAD"], capture_output=True, text=True)

    diff_text = diff_res.stdout.strip()
    if not diff_text:
        console.print("[yellow]No modifications detected in Git repository to review.[/yellow]")
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
    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_model}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    system_prompt = (
        "You are a strict, automated AI code reviewer. Analyze the following Git diff for code quality, "
        "logic errors, security vulnerabilities (like secrets exposure, SQL injection), and code smells.\n\n"
        "Your output must be in JSON format containing a list of issues under the key \"issues\". "
        "Each issue must have the following keys:\n"
        '- "file": The file path containing the issue.\n'
        '- "line": The line number or approximate line range.\n'
        '- "severity": One of "critical" (security risk, crash bug, secrets leak) or "warning" (code smell, formatting, minor bug).\n'
        '- "description": A clear, concise explanation of the issue and why it is problematic.\n'
        '- "suggestion": Code recommendation or fix.\n\n'
        "If no issues are found, return an empty list under \"issues\".\n"
        "Output strictly valid JSON. Do not wrap in markdown code blocks."
    )

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the diff:\n\n{diff_text}"}
        ],
        "stream": False
    }

    console.print(f"Auditing code modifications using [bold cyan]{chosen_model}[/bold cyan]...")
    try:
        response = httpx.post(url_chat, json=payload, timeout=180.0)
        if response.status_code != 200:
            console.print(f"[red]Failed to generate review audit: {response.text}[/red]")
            raise typer.Exit(1)
        result = response.json()
        raw_text = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        console.print(f"[red]Error contacting Gateway: {e}[/red]")
        raise typer.Exit(1)

    # Clean markdown wrapping if present
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        if isinstance(data, list):
            issues = data
        elif isinstance(data, dict):
            issues = data.get("issues", [])
        else:
            issues = []
    except Exception:
        console.print(f"[red]Error: Failed to parse generated review as JSON. Raw output was:[/red]\n{raw_text}")
        raise typer.Exit(1)

    if not issues:
        console.print("[bold green]All checks passed! No issues detected in your modifications.[/bold green]")
        return

    # Count issues
    criticals = [i for i in issues if i.get("severity", "").lower() == "critical"]
    warnings = [i for i in issues if i.get("severity", "").lower() != "critical"]

    console.print(f"\n[bold white]Review Audit Summary: Found {len(criticals)} critical issue(s) and {len(warnings)} warning(s).[/bold white]\n")

    from rich.panel import Panel
    from rich.console import Group

    for issue in issues:
        file_path = issue.get("file", "Unknown")
        line = issue.get("line", "-")
        severity = issue.get("severity", "warning").upper()
        desc = issue.get("description", "")
        suggestion = issue.get("suggestion", "")

        border_style = "red" if severity.lower() == "critical" else "yellow"
        title = f"[{border_style}]{severity}[/{border_style}] — {file_path}:{line}"

        content_group = Group(
            f"[bold white]Description:[/bold white] {desc}\n",
            f"[bold white]Suggestion:[/bold white]\n  {suggestion}"
        )

        console.print(Panel(
            content_group,
            title=title,
            border_style=border_style,
            expand=False
        ))

    if pre_commit and criticals:
        console.print("\n[bold red]Commit rejected: Critical issues found in staged files.[/bold red]")
        raise typer.Exit(1)
