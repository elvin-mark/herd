import os
import json
import subprocess
import shutil
import httpx
import typer
import asyncio
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


def heal(
    ctx: typer.Context,
    command: Optional[str] = typer.Argument(None, help="The command string to execute (e.g. 'python3 script.py')."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model identifier to diagnose the issue.",
    ),
):
    """Executes a shell command and automatically diagnoses/heals it using local LLMs on failure."""
    command_parts = []
    if command:
        command_parts.append(command)
    if ctx.args:
        command_parts.extend(ctx.args)

    if not command_parts:
        console.print("[red]Error: Please specify a command to execute.[/red]")
        raise typer.Exit(1)

    command_str = " ".join(command_parts)

    console.print(f"[bold cyan]Executing command:[/bold cyan] {command_str}\n")

    # Run the command and capture logs in real-time
    output_buffer = []
    try:
        process = subprocess.Popen(
            command_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end="")
            output_buffer.append(line)
            if len(output_buffer) > 100:  # keep last 100 lines
                output_buffer.pop(0)
        process.wait()
        exit_code = process.returncode
    except Exception as e:
        console.print(f"\n[red]Failed to execute subprocess: {e}[/red]")
        raise typer.Exit(1)

    if exit_code == 0:
        console.print("\n[bold green]Command completed successfully (exit code 0).[/bold green]")
        return

    console.print(f"\n[bold red]⚠️ Command failed with exit code {exit_code}. Analyzing failure logs...[/bold red]")

    # Resolve LLM model
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

    logs_text = "".join(output_buffer)

    system_prompt = (
        "You are an expert systems and operations debugging assistant. Diagnose why the user's terminal command failed "
        "given the command string and the execution logs (stdout/stderr trace).\n\n"
        "Your response must be in JSON format with exactly three keys:\n"
        "1. \"error_explanation\": A clear, concise (1-2 sentences) explanation of what caused the crash.\n"
        "2. \"suggested_fix\": The single terminal command or action to run that will fix the error (e.g. `pip install numpy`, `chmod +x script.sh`, or correct parameters/syntax).\n"
        "3. \"can_auto_run\": A boolean indicating if Herd can automatically execute this suggested fix command on confirmation (set to true ONLY if it is a safe command-line utility execution like installing a package, changing permissions, creating a folder, or running a clean syntax command; set to false if it requires manual file edits or unsafe actions).\n\n"
        "Do not output markdown code blocks. Output strictly valid JSON."
    )

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Command: {command_str}\n\nExecution Logs:\n{logs_text}"}
        ],
        "stream": False
    }

    try:
        response = httpx.post(url_chat, json=payload, timeout=45.0)
        if response.status_code != 200:
            console.print(f"[red]Failed to analyze logs: {response.text}[/red]")
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
        explanation = data["error_explanation"]
        suggested_fix = data["suggested_fix"]
        can_auto_run = data["can_auto_run"]
    except Exception:
        console.print(f"[red]Error: Failed to parse diagnosis response as JSON. Raw output was:[/red]\n{raw_text}")
        raise typer.Exit(1)

    # Print diagnosis panel
    from rich.panel import Panel
    from rich.console import Group
    panel_group = Group(
        f"[bold white]Diagnosis:[/bold white]\n  {explanation}\n",
        f"[bold white]Suggested Fix:[/bold white]\n  [bold yellow]{suggested_fix}[/bold yellow]"
    )
    console.print(Panel(
        panel_group,
        title="[bold red]Herd Self-Healing System Diagnosis[/bold red]",
        border_style="red",
        expand=False
    ))

    if can_auto_run and suggested_fix:
        confirm_fix = typer.confirm(f"\nWould you like to execute the suggested fix command: {suggested_fix}?")
        if confirm_fix:
            console.print(f"\n[bold cyan]Executing fix:[/bold cyan] {suggested_fix}\n")
            try:
                subprocess.run(suggested_fix, shell=True, check=True)
                console.print("[bold green]Fix executed successfully![/bold green]")

                # Ask to rerun original command
                confirm_rerun = typer.confirm(f"\nWould you like to re-run the original command: {command_str}?")
                if confirm_rerun:
                    console.print(f"\n[bold cyan]Re-running original command:[/bold cyan] {command_str}\n")
                    subprocess.run(command_str, shell=True)
            except Exception as e:
                console.print(f"[red]Failed to execute fix or original command: {e}[/red]")
    else:
        console.print("\n[yellow]This issue requires manual intervention or file editing. Please apply the fix above manually.[/yellow]")


async def stream_watch_async(model_name: str, image_data: str, prompt: str):
    """Sends a streaming chat completion request with vision multimodal payload."""
    url = f"{get_gateway_url()}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data
                        }
                    }
                ]
            }
        ],
        "stream": True
    }

    console.print(f"\n[bold green]Querying multimodal model {model_name}...[/bold green]\n")
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code != 200:
                err_body = await response.aread()
                console.print(f"[red]Request failed: {err_body.decode()}[/red]")
                raise typer.Exit(1)
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                lines = chunk.decode("utf-8", errors="ignore").split("\n")
                for line in lines:
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            token = data["choices"][0]["delta"].get("content", "")
                            if token is not None:
                                print(token, end="", flush=True)
                        except Exception:
                            pass
    print("\n")


def watch(
    image_path: str = typer.Argument(..., help="Path to local image file (or URL) to analyze."),
    prompt: str = typer.Argument("Describe the image.", help="The prompt/question to ask the model about the image."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="VLM model identifier to run.",
    ),
):
    """Analyzes visual inputs (images, URLs) using multimodal vision-language models (VLM)."""
    # 1. Encode image
    import base64
    import mimetypes

    if image_path.startswith("http://") or image_path.startswith("https://"):
        console.print(f"Downloading image from [bold cyan]{image_path}[/bold cyan]...")
        try:
            res = httpx.get(image_path)
            res.raise_for_status()
            mime_type = res.headers.get("content-type", "image/jpeg")
            encoded_string = base64.b64encode(res.content).decode('utf-8')
            image_data = f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            console.print(f"[red]Error downloading image: {e}[/red]")
            raise typer.Exit(1)
    else:
        if not os.path.exists(image_path):
            console.print(f"[red]Error: Image file not found: {image_path}[/red]")
            raise typer.Exit(1)
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            image_data = f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            console.print(f"[red]Error reading image: {e}[/red]")
            raise typer.Exit(1)

    # 2. Resolve VLM
    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No VLM models found. Please pull a model first.[/red]")
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

    # 3. Stream watch query
    try:
        asyncio.run(stream_watch_async(chosen_model, image_data, prompt))
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation stopped.[/yellow]")


def agent_list_dir(path: str = ".") -> str:
    try:
        files = os.listdir(path)
        return json.dumps(files)
    except Exception as e:
        return f"Error listing directory: {e}"


def agent_read_file(path: str) -> str:
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


def agent_write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"File successfully written to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def agent_run_command(command: str) -> str:
    # Run command and capture output
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30.0)
        output = f"Exit Code: {res.returncode}\n"
        if res.stdout:
            output += f"STDOUT:\n{res.stdout}\n"
        if res.stderr:
            output += f"STDERR:\n{res.stderr}\n"
        return output
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out (30s limit)."
    except Exception as e:
        return f"Error executing command: {e}"


def agent(
    objective: str = typer.Argument(..., help="The objective/task for the agent to accomplish (e.g. 'find todos in python files')."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="LLM model identifier to run.",
    ),
    max_turns: int = typer.Option(
        10,
        "--max-turns",
        "-t",
        help="Maximum execution turns/iterations to run.",
    ),
):
    """Launches an autonomous local AI agent loop to execute multi-step tasks in your workspace."""
    # 1. Resolve LLM model
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
        "You are an autonomous AI engineering agent executing tasks in a local workspace.\n"
        "You operate in a loop: Thought -> Action -> Observation -> Repeat.\n"
        "Your goal is to satisfy the user's objective.\n\n"
        "Available Tools:\n"
        "1. list_dir: List files in a folder. Action Input should be the folder path (e.g. '.' or './src').\n"
        "2. read_file: Read a text file. Action Input should be the path to the file.\n"
        "3. write_file: Write/overwrite a file. Action Input should be a JSON object containing \"path\" and \"content\".\n"
        "4. run_command: Run a shell command. Action Input should be the command string.\n"
        "5. final_answer: Signal that you have finished the objective. Action Input should be a summary of the result.\n\n"
        "At each turn, you MUST output a valid JSON object matching the following structure:\n"
        "{\n"
        "  \"thought\": \"What you are planning to do and why\",\n"
        "  \"action\": \"The tool name to call (list_dir, read_file, write_file, run_command, final_answer)\",\n"
        "  \"action_input\": \"The raw parameter string or JSON payload required by the tool\"\n"
        "}\n\n"
        "Remember:\n"
        "- Do not explain your response outside of the JSON object.\n"
        "- Output strictly valid JSON. Do not wrap in markdown code blocks."
    )

    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Objective: {objective}"}
    ]

    console.print("\n🚀 [bold green]Starting Autonomous Agent Loop[/bold green]")
    console.print(f"  Model:     [bold cyan]{chosen_model}[/bold cyan]")
    console.print(f"  Objective: [bold yellow]{objective}[/bold yellow]\n")

    url_chat = f"{get_gateway_url()}/v1/chat/completions"

    for turn in range(1, max_turns + 1):
        console.print(f"[bold dim]── Turn {turn}/{max_turns} ──────────────────────────────────────[/bold dim]")

        # 1. Ask LLM for next step
        try:
            res = httpx.post(url_chat, json={
                "model": chosen_model,
                "messages": history,
                "temperature": 0.2,
                "stream": False
            }, timeout=60.0)
            if res.status_code != 200:
                console.print(f"[red]Error from Gateway: {res.text}[/red]")
                break
            raw_text = res.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            console.print(f"[red]Error communicating with LLM: {e}[/red]")
            break

        # Cleanup markdown syntax if returned
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()

        # 2. Parse action JSON
        try:
            action_data = json.loads(raw_text)
            thought = action_data["thought"]
            action = action_data["action"]
            action_input = action_data["action_input"]
        except Exception:
            console.print(f"[red]Error: Model output was not valid JSON. Raw output was:[/red]\n{raw_text}")
            history.append({"role": "assistant", "content": raw_text})
            history.append({"role": "user", "content": "Please output strictly a valid JSON object containing 'thought', 'action', and 'action_input'."})
            continue

        # Print thought
        console.print(Panel(
            f"[italic white]{thought}[/italic white]",
            title=f"🧠 Agent Thought (Turn {turn})",
            border_style="cyan"
        ))

        # 3. Handle Actions
        if action == "final_answer":
            console.print(Panel(
                f"[bold green]Final Answer:[/bold green]\n{action_input}",
                title="🏁 Objective Accomplished",
                border_style="green"
            ))
            break

        console.print(f"⚙️  [bold]Action:[/bold] {action} | [bold]Input:[/bold] {action_input}")

        observation = ""
        if action == "list_dir":
            observation = agent_list_dir(action_input or ".")
        elif action == "read_file":
            observation = agent_read_file(action_input)
        elif action == "write_file":
            try:
                if isinstance(action_input, str):
                    write_data = json.loads(action_input)
                else:
                    write_data = action_input
                observation = agent_write_file(write_data["path"], write_data["content"])
            except Exception as e:
                observation = f"Error parsing write_file parameters: {e}. Expected a JSON object with 'path' and 'content'."
        elif action == "run_command":
            observation = agent_run_command(action_input)
        else:
            observation = f"Error: Unknown action '{action}'."

        # Show observation
        console.print(f"👁️  [bold]Observation:[/bold] {observation[:400]}..." if len(observation) > 400 else f"👁️  [bold]Observation:[/bold] {observation}")

        history.append({"role": "assistant", "content": raw_text})
        history.append({"role": "user", "content": f"Observation: {observation}"})
