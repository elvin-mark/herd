import asyncio
import json
import os
import shutil
import subprocess
from typing import Optional

import httpx
import typer
from rich.console import Group
from rich.panel import Panel

from herd.core.utils import (
    auto_start_gateway,
    console,
    extract_reasoning_and_json,
    find_running_llm,
    get_gateway_url,
)


def copilot(
    instruction: str = typer.Argument(
        ...,
        help="Natural language prompt describing what you want to execute in the terminal.",
    ),
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
            {"role": "user", "content": instruction},
        ],
        "stream": False,
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

    # Extract think block if present and format JSON text
    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    try:
        data = json.loads(cleaned_text)
        command = data["command"]
        explanation = data["explanation"]
    except Exception:
        console.print(
            f"[red]Error: Failed to parse generated response as JSON. Raw output was:[/red]\n{raw_text}"
        )
        raise typer.Exit(1)

    # Print proposal
    panel_group = Group(
        f"[bold white]Command:[/bold white]\n  [bold green]{command}[/bold green]\n",
        f"[bold white]Explanation:[/bold white]\n  {explanation}",
    )
    console.print(
        Panel(
            panel_group,
            title="[bold cyan]Herd Shell Copilot Proposal[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

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
        console.print(
            "[yellow]Warning: Git diff is very large. Truncating to 10,000 characters.[/yellow]"
        )
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
            {"role": "user", "content": f"Here is the diff:\n\n{diff_text}"},
        ],
        "stream": False,
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

    # Intercept and neatly extract any reasoning/thinking blocks
    think_content, cleaned_text = extract_reasoning_and_json(commit_message)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    commit_message = cleaned_text.strip()

    # Clean markdown wrapping if present
    if commit_message.startswith("```"):
        lines = commit_message.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        commit_message = "\n".join(lines).strip()

    console.print(
        Panel(
            commit_message,
            title="[bold cyan]Proposed Conventional Commit Message[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

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
        help="Install herd git review as a local Git pre-commit hook.",
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

        script = "#!/bin/sh\nherd git review --pre-commit\n"
        try:
            with open(hook_path, "w") as f:
                f.write(script)
            os.chmod(hook_path, 0o755)
            console.print(
                "[bold green]Success![/bold green] Installed pre-commit hook at .git/hooks/pre-commit"
            )
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
        console.print(
            "[yellow]Warning: Git diff is very large. Truncating to 10,000 characters.[/yellow]"
        )
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
        'Your output must be in JSON format containing a list of issues under the key "issues". '
        "Each issue must have the following keys:\n"
        '- "file": The file path containing the issue.\n'
        '- "line": The line number or approximate line range.\n'
        '- "severity": One of "critical" (security risk, crash bug, secrets leak) or "warning" (code smell, formatting, minor bug).\n'
        '- "description": A clear, concise explanation of the issue and why it is problematic.\n'
        '- "suggestion": Code recommendation or fix.\n\n'
        'If no issues are found, return an empty list under "issues".\n'
        "Output strictly valid JSON. Do not wrap in markdown code blocks."
    )

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the diff:\n\n{diff_text}"},
        ],
        "stream": False,
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

    # Extract think block if present and format JSON text
    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    try:
        data = json.loads(cleaned_text)
        if isinstance(data, list):
            issues = data
        elif isinstance(data, dict):
            issues = data.get("issues", [])
        else:
            issues = []
    except Exception:
        console.print(
            f"[red]Error: Failed to parse generated review as JSON. Raw output was:[/red]\n{raw_text}"
        )
        raise typer.Exit(1)

    if not issues:
        console.print(
            "[bold green]All checks passed! No issues detected in your modifications.[/bold green]"
        )
        return

    # Count issues
    criticals = [i for i in issues if i.get("severity", "").lower() == "critical"]
    warnings = [i for i in issues if i.get("severity", "").lower() != "critical"]

    console.print(
        f"\n[bold white]Review Audit Summary: Found {len(criticals)} critical issue(s) and {len(warnings)} warning(s).[/bold white]\n"
    )

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
            f"[bold white]Suggestion:[/bold white]\n  {suggestion}",
        )

        console.print(Panel(content_group, title=title, border_style=border_style, expand=False))

    if pre_commit and criticals:
        console.print(
            "\n[bold red]Commit rejected: Critical issues found in staged files.[/bold red]"
        )
        raise typer.Exit(1)


def heal(
    ctx: typer.Context,
    command: Optional[str] = typer.Argument(
        None, help="The command string to execute (e.g. 'python3 script.py')."
    ),
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
            bufsize=1,
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

    console.print(
        f"\n[bold red]⚠️ Command failed with exit code {exit_code}. Analyzing failure logs...[/bold red]"
    )

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
        '1. "error_explanation": A clear, concise (1-2 sentences) explanation of what caused the crash.\n'
        '2. "suggested_fix": The single terminal command or action to run that will fix the error (e.g. `pip install numpy`, `chmod +x script.sh`, or correct parameters/syntax).\n'
        '3. "can_auto_run": A boolean indicating if Herd can automatically execute this suggested fix command on confirmation (set to true ONLY if it is a safe command-line utility execution like installing a package, changing permissions, creating a folder, or running a clean syntax command; set to false if it requires manual file edits or unsafe actions).\n\n'
        "Do not output markdown code blocks. Output strictly valid JSON."
    )

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Command: {command_str}\n\nExecution Logs:\n{logs_text}",
            },
        ],
        "stream": False,
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

    # Extract think block if present and format JSON text
    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    try:
        data = json.loads(cleaned_text)
        explanation = data["error_explanation"]
        suggested_fix = data["suggested_fix"]
        can_auto_run = data["can_auto_run"]
    except Exception:
        console.print(
            f"[red]Error: Failed to parse diagnosis response as JSON. Raw output was:[/red]\n{raw_text}"
        )
        raise typer.Exit(1)

    # Print diagnosis panel
    panel_group = Group(
        f"[bold white]Diagnosis:[/bold white]\n  {explanation}\n",
        f"[bold white]Suggested Fix:[/bold white]\n  [bold yellow]{suggested_fix}[/bold yellow]",
    )
    console.print(
        Panel(
            panel_group,
            title="[bold red]Herd Self-Healing System Diagnosis[/bold red]",
            border_style="red",
            expand=False,
        )
    )

    if can_auto_run and suggested_fix:
        confirm_fix = typer.confirm(
            f"\nWould you like to execute the suggested fix command: {suggested_fix}?"
        )
        if confirm_fix:
            console.print(f"\n[bold cyan]Executing fix:[/bold cyan] {suggested_fix}\n")
            try:
                subprocess.run(suggested_fix, shell=True, check=True)
                console.print("[bold green]Fix executed successfully![/bold green]")

                # Ask to rerun original command
                confirm_rerun = typer.confirm(
                    f"\nWould you like to re-run the original command: {command_str}?"
                )
                if confirm_rerun:
                    console.print(
                        f"\n[bold cyan]Re-running original command:[/bold cyan] {command_str}\n"
                    )
                    subprocess.run(command_str, shell=True)
            except Exception as e:
                console.print(f"[red]Failed to execute fix or original command: {e}[/red]")
    else:
        console.print(
            "\n[yellow]This issue requires manual intervention or file editing. Please apply the fix above manually.[/yellow]"
        )


async def stream_watch_async(model_name: str, image_data: str, prompt: str):
    """Sends a streaming chat completion request with vision multimodal payload."""
    url = f"{get_gateway_url()}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            }
        ],
        "stream": True,
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


def vision(
    image_path: str = typer.Argument(..., help="Path to local image file (or URL) to analyze."),
    prompt: str = typer.Argument(
        "Describe the image.",
        help="The prompt/question to ask the model about the image.",
    ),
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
            encoded_string = base64.b64encode(res.content).decode("utf-8")
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
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
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


def agent(
    objective: Optional[str] = typer.Argument(
        None,
        help="Initial objective/task for the agent to accomplish. If omitted, starts an interactive session.",
    ),
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
        help="Maximum execution turns/iterations to run per request.",
    ),
    yolo: bool = typer.Option(
        False,
        "--yolo",
        "-y",
        help="Bypass interactive approvals for running shell commands.",
    ),
    memory: bool = typer.Option(
        True,
        "--memory/--no-memory",
        help="Enable episodic long-term memory for the agent.",
    ),
):
    """Launches an interactive or autonomous local AI agent loop to execute multi-step tasks in your workspace."""
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

    # Instantiate the agent session
    from herd.services.agent import AgentSession

    session = AgentSession(chosen_model, get_gateway_url(), yolo=yolo, use_memory=memory)

    console.print("\n🚀 [bold green]Starting Herd Agent Interface[/bold green]")
    console.print(f"  Model: [bold cyan]{chosen_model}[/bold cyan]")
    console.print(
        "  Type [bold red]exit[/bold red] or [bold red]quit[/bold red] to end the session.\n"
    )

    # If an initial objective was passed, run it and exit!
    if objective:
        console.print(f"\n[bold yellow]Objective:[/bold yellow] {objective}")
        session.run_task(objective, max_turns=max_turns)
        return

    # Chat REPL loop
    while True:
        try:
            user_input = console.input("[bold green]Agent 🤖 ❯ [/bold green]").strip()
            if not user_input:
                continue

            # Built-in Slash Commands
            if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
                console.print("[yellow]Exiting agent session. Goodbye![/yellow]")
                break

            if user_input.lower() == "/help":
                console.print("\n[bold cyan]Herd Agent Commands:[/bold cyan]")
                console.print("  [bold white]/help[/bold white]   - Show this help message")
                console.print("  [bold white]/exit[/bold white]   - Exit the agent session")
                console.print(
                    "  [bold white]/show[/bold white]   - Show current session configuration and active model"
                )
                console.print(
                    "  [bold white]/tools[/bold white]  - List all registered agent tools"
                )
                console.print(
                    "  [bold white]/usage[/bold white]  - Show estimated token context usage"
                )
                console.print(
                    "  [bold white]/yolo[/bold white]   - Toggle YOLO mode (auto-execute shell commands) ON/OFF"
                )
                console.print("  [bold white]/memory[/bold white] - Toggle long-term memory ON/OFF")
                console.print(
                    "  [bold white]/clear[/bold white]  - Clear the conversational history (resets context)"
                )
                console.print(
                    "  [bold white]/save[/bold white]   - Save the current conversation history to a file"
                )
                console.print(
                    "  [bold white]/system[/bold white] - Show the current agent system prompt\n"
                )
                continue

            if user_input.lower() in ("/show", "/info"):
                from rich.table import Table

                table = Table(title="Agent Session Configuration", show_header=False, box=None)
                table.add_row(
                    "[bold cyan]Model:[/bold cyan]",
                    f"[white]{session.model_name}[/white]",
                )
                table.add_row(
                    "[bold cyan]Gateway:[/bold cyan]",
                    f"[white]{session.gateway_url}[/white]",
                )
                table.add_row(
                    "[bold cyan]YOLO Mode:[/bold cyan]",
                    "[bold green]ON[/bold green]"
                    if session.yolo
                    else "[bold yellow]OFF[/bold yellow]",
                )
                table.add_row(
                    "[bold cyan]Memory:[/bold cyan]",
                    "[bold green]ON[/bold green]"
                    if session.use_memory
                    else "[bold yellow]OFF[/bold yellow]",
                )
                table.add_row(
                    "[bold cyan]Tools Loaded:[/bold cyan]",
                    f"[white]{len(session.registry.tools)}[/white]",
                )
                console.print()
                console.print(table)
                console.print()
                continue

            if user_input.lower() == "/tools":
                from rich.table import Table

                table = Table(title="Registered Agent Tools", header_style="cyan")
                table.add_column("Tool Name", style="bold white")
                table.add_column("Description", style="dim white")
                for name, tool in session.registry.tools.items():
                    table.add_row(name, tool.description)
                console.print()
                console.print(table)
                console.print()
                continue

            if user_input.lower() == "/yolo":
                session.yolo = not session.yolo
                state = (
                    "[bold green]ON[/bold green] (Auto-executing shell commands)"
                    if session.yolo
                    else "[bold yellow]OFF[/bold yellow] (Requiring manual approval)"
                )
                console.print(f"\n[bold cyan]YOLO Mode is now {state}[/bold cyan]\n")
                continue

            if user_input.lower() == "/memory":
                session.use_memory = not session.use_memory
                state = (
                    "[bold green]ON[/bold green]"
                    if session.use_memory
                    else "[bold yellow]OFF[/bold yellow]"
                )
                console.print(f"\n[bold cyan]Long-Term Memory is now {state}[/bold cyan]\n")
                continue

            if user_input.lower() == "/save":
                import time

                filename = f"agent_transcript_{int(time.time())}.md"
                with open(filename, "w") as f:
                    f.write(
                        f"# Herd Agent Transcript\nModel: {session.model_name}\nDate: {time.ctime()}\n\n"
                    )
                    for msg in session.history:
                        role = msg.get("role", "unknown").upper()
                        f.write(f"### {role}\n```\n{msg.get('content', '')}\n```\n\n")
                console.print(f"\n[bold green]✓ Transcript saved to {filename}[/bold green]\n")
                continue

            if user_input.lower() == "/usage":
                # Roughly estimate 1 token per 4 characters
                approx_tokens = sum(len(str(m.get("content", ""))) // 4 for m in session.history)
                console.print("\n[bold green]Estimated Session Usage:[/bold green]")
                console.print(f"  Turns / Messages: [white]{len(session.history)}[/white]")
                console.print(f"  Current Context Load: [white]~{approx_tokens:,} tokens[/white]\n")
                continue

            if user_input.lower() == "/clear":
                session.history = [{"role": "system", "content": session.system_prompt}]
                console.print(
                    "\n[bold green]✓ Session history cleared. Context is fresh.[/bold green]\n"
                )
                continue

            if user_input.lower() == "/system":
                from rich.panel import Panel

                console.print(
                    Panel(
                        session.system_prompt,
                        title="[cyan]Current System Prompt[/cyan]",
                        border_style="cyan",
                    )
                )
                continue

            session.run_task(user_input, max_turns=max_turns)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Exiting agent session. Goodbye![/yellow]")
            break


def triage(
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="Specify the model to use."
    ),
):
    """Analyze uncommitted changes and group them into logical, atomic commits."""
    # Check if we are in a git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError:
        console.print("[red]Error: Not a git repository. Please run inside a git project.[/red]")
        raise typer.Exit(1)

    # Get unstaged changes (or staged if empty)
    diff_res = subprocess.run(
        ["git", "diff"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    diff_text = diff_res.stdout.strip()

    if not diff_text:
        diff_res = subprocess.run(
            ["git", "diff", "--cached"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        diff_text = diff_res.stdout.strip()

    if not diff_text:
        console.print("[yellow]No uncommitted or staged changes detected to triage.[/yellow]")
        return

    if len(diff_text) > 10000:
        console.print(
            "[yellow]Warning: Git diff is very large. Truncating to 10,000 characters.[/yellow]"
        )
        diff_text = diff_text[:10000] + "\n\n... [TRUNCATED] ..."

    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No local LLM models found. Please pull a model first.[/red]")
        raise typer.Exit(1)

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
        "You are an expert software engineer. Analyze the following Git diff and group the modified "
        "files into logical, independent atomic commits based on their semantic purpose (e.g., separating "
        "bug fixes from new features or refactors).\n\n"
        'Your output MUST be valid JSON containing a single key "commits", which is a list of objects. '
        "Each object must have the following keys:\n"
        '- "theme": A conventional commit message title for this group (e.g., "fix: resolve port conflict").\n'
        '- "files": A list of file paths that belong to this atomic change.\n'
        '- "reasoning": A 1-2 sentence explanation of why these files belong together.\n\n'
        "Do not wrap the output in markdown code blocks. Output strictly raw JSON."
    )

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the diff:\n\n{diff_text}"},
        ],
        "stream": False,
    }

    console.print(f"Analyzing diff semantics using [bold cyan]{chosen_model}[/bold cyan]...")
    try:
        response = httpx.post(url_chat, json=payload, timeout=120.0)
        if response.status_code != 200:
            console.print(f"[red]Failed to generate triage plan: {response.text}[/red]")
            raise typer.Exit(1)
        result = response.json()
        raw_text = result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        console.print(f"[red]Error contacting Gateway: {e}[/red]")
        raise typer.Exit(1)

    # Re-use our centralized extraction logic
    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    try:
        data = json.loads(cleaned_text)
        commits = data.get("commits", [])
    except Exception:
        console.print(
            f"[red]Error: Failed to parse generated triage plan as JSON. Raw output was:[/red]\n{raw_text}"
        )
        raise typer.Exit(1)

    if not commits:
        console.print(
            "[yellow]Could not logically group the changes. They might be too entangled.[/yellow]"
        )
        return

    console.print("\n[bold green]Recommended Atomic Commits:[/bold green]\n")
    for idx, commit_plan in enumerate(commits, 1):
        console.print(
            f"📦 [bold magenta]Commit {idx}: {commit_plan.get('theme', 'Updates')}[/bold magenta]"
        )
        console.print(f"   [dim]{commit_plan.get('reasoning', '')}[/dim]")
        for file in commit_plan.get("files", []):
            console.print(f"   - [cyan]{file}[/cyan]")

        # Output git command hint
        file_args = " ".join([f'"{f}"' for f in commit_plan.get("files", [])])
        console.print(f"   [dim yellow]> git add {file_args}[/dim yellow]\n")
    print("\n")


def pr(
    base: str = typer.Option(
        "main",
        "--base",
        "-b",
        help="The base branch to compare against (e.g., main or master).",
    ),
    model_name: Optional[str] = typer.Option(
        None, "--model", "-m", help="Specify the model to use."
    ),
):
    """Generate a comprehensive Markdown Pull Request description based on branch history and code diffs."""
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError:
        console.print("[red]Error: Not a git repository. Please run inside a git project.[/red]")
        raise typer.Exit(1)

    # 1. Get commit history
    log_res = subprocess.run(
        ["git", "log", f"{base}..HEAD", "--pretty=format:%h - %s"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if log_res.returncode != 0:
        console.print(f"[red]Error comparing branches: Could not find base branch '{base}'.[/red]")
        raise typer.Exit(1)

    commit_history = log_res.stdout.strip()

    # 2. Get code diff
    diff_res = subprocess.run(
        ["git", "diff", f"{base}...HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    diff_text = diff_res.stdout.strip()

    if not diff_text and not commit_history:
        console.print(f"[yellow]No differences found between '{base}' and HEAD.[/yellow]")
        return

    # Truncate to save tokens
    if len(diff_text) > 15000:
        console.print(
            "[yellow]Warning: Git diff is very large. Truncating to 15,000 characters.[/yellow]"
        )
        diff_text = diff_text[:15000] + "\n\n... [TRUNCATED] ..."

    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No local LLM models found.[/red]")
        raise typer.Exit(1)

    if not auto_start_gateway():
        raise typer.Exit(1)

    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_model}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    system_prompt = (
        "You are an expert technical writer and senior software engineer. "
        "Your task is to draft a comprehensive, professional Markdown Pull Request description.\n\n"
        "You will be provided with:\n"
        "1. The COMMIT HISTORY of the feature branch (to understand the developer's intent and steps).\n"
        "2. The consolidated CODE DIFF against the base branch.\n\n"
        "Your Pull Request should include:\n"
        "- A clear, engaging Title (Header 1).\n"
        "- A 'Summary' section explaining the high-level purpose of the PR.\n"
        "- A 'Key Changes' section with bullet points.\n"
        "- A 'Reasoning / Context' section if applicable.\n"
        "- Do NOT output any JSON. Output pure Markdown."
    )

    user_prompt = f"### COMMIT HISTORY:\n{commit_history}\n\n### CODE DIFF:\n{diff_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {"model": chosen_model, "messages": messages, "stream": False}

    with console.status(
        f"[bold cyan]Generating PR Description using {chosen_model}...[/bold cyan]",
        spinner="dots",
    ):
        try:
            response = httpx.post(url_chat, json=payload, timeout=180.0)
            if response.status_code != 200:
                console.print(f"[red]Failed: {response.text}[/red]")
                raise typer.Exit(1)
            raw_text = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            console.print(f"[red]Error contacting Gateway: {e}[/red]")
            raise typer.Exit(1)

    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    # Strip outer markdown block if the model wrapped the entire response
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

    from rich.markdown import Markdown

    console.print(Markdown(cleaned_text))


def test_cmd(
    filename: str = typer.Argument(..., help="Path to the file to generate tests for."),
    model_name: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Autonomous Test Scaffolding: Generates unit tests for the specified file."""
    if not os.path.exists(filename):
        console.print(f"[red]Error: File {filename} not found.[/red]")
        raise typer.Exit(1)

    with open(filename, "r") as f:
        code_content = f.read()

    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No local LLM models found.[/red]")
        raise typer.Exit(1)

    if not auto_start_gateway():
        raise typer.Exit(1)

    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_model}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    system_prompt = (
        "You are an expert software engineer specializing in testing. "
        "Your task is to write comprehensive unit tests for the provided code.\n"
        "Use modern testing frameworks (e.g., pytest for Python, jest for JS/TS).\n"
        "Include edge cases and mock external dependencies if necessary.\n"
        "Output ONLY the raw test code. Do not include markdown code blocks or explanations."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"File content for {filename}:\n\n{code_content}"},
    ]

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {"model": chosen_model, "messages": messages, "stream": False}

    with console.status(
        f"[bold cyan]Generating tests for {filename} using {chosen_model}...[/bold cyan]",
        spinner="dots",
    ):
        try:
            response = httpx.post(url_chat, json=payload, timeout=180.0)
            if response.status_code != 200:
                console.print(f"[red]Failed: {response.text}[/red]")
                raise typer.Exit(1)
            raw_text = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            console.print(f"[red]Error contacting Gateway: {e}[/red]")
            raise typer.Exit(1)

    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    # Clean markdown if present
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

    from rich.syntax import Syntax

    console.print(Syntax(cleaned_text, "python", theme="monokai", line_numbers=True))

    confirm = typer.confirm("\nWould you like to save these tests to a file?")
    if confirm:
        test_filename = typer.prompt("Enter filename", default=f"test_{os.path.basename(filename)}")
        with open(test_filename, "w") as f:
            f.write(cleaned_text)
        console.print(f"[bold green]Successfully saved tests to {test_filename}![/bold green]")
    else:
        console.print("[yellow]Aborted.[/yellow]")


def docs_cmd(
    filename: str = typer.Argument(..., help="Path to the file to document."),
    model_name: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Inline Documentation Generator: Adds docstrings to undocumented functions/classes."""
    if not os.path.exists(filename):
        console.print(f"[red]Error: File {filename} not found.[/red]")
        raise typer.Exit(1)

    with open(filename, "r") as f:
        code_content = f.read()

    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No local LLM models found.[/red]")
        raise typer.Exit(1)

    if not auto_start_gateway():
        raise typer.Exit(1)

    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_model}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    system_prompt = (
        "You are an expert technical writer and developer. "
        "Your task is to read the following code and output the EXACT same code, but with "
        "comprehensive, standard docstrings (e.g., Google style for Python, JSDoc for JS/TS) added to every function and class.\n"
        "Do NOT change the logic of the code. Output ONLY the raw code. Do not include markdown code blocks or explanations."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"File content for {filename}:\n\n{code_content}"},
    ]

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {"model": chosen_model, "messages": messages, "stream": False}

    with console.status(
        f"[bold cyan]Generating documentation for {filename} using {chosen_model}...[/bold cyan]",
        spinner="dots",
    ):
        try:
            response = httpx.post(url_chat, json=payload, timeout=180.0)
            if response.status_code != 200:
                console.print(f"[red]Failed: {response.text}[/red]")
                raise typer.Exit(1)
            raw_text = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            console.print(f"[red]Error contacting Gateway: {e}[/red]")
            raise typer.Exit(1)

    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    # Clean markdown if present
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

    from rich.syntax import Syntax

    console.print(Syntax(cleaned_text, "python", theme="monokai", line_numbers=True))

    confirm = typer.confirm(f"\nWould you like to overwrite {filename} with these changes?")
    if confirm:
        with open(filename, "w") as f:
            f.write(cleaned_text)
        console.print(f"[bold green]Successfully updated {filename}![/bold green]")
    else:
        console.print("[yellow]Aborted.[/yellow]")


def refactor_cmd(
    filename: str = typer.Argument(..., help="Path to the file to refactor."),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="Instructions for the refactor (e.g., 'Add type hints').",
    ),
    model_name: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Semantic File Rewriter: Applies a custom semantic transformation to a file."""
    if not os.path.exists(filename):
        console.print(f"[red]Error: File {filename} not found.[/red]")
        raise typer.Exit(1)

    with open(filename, "r") as f:
        code_content = f.read()

    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No local LLM models found.[/red]")
        raise typer.Exit(1)

    if not auto_start_gateway():
        raise typer.Exit(1)

    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_model}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    system_prompt = (
        "You are an expert senior software engineer. "
        "Your task is to refactor the provided code according to the user's specific instructions.\n"
        "Ensure the code remains functional and follows best practices.\n"
        "Output ONLY the raw refactored code. Do not include markdown code blocks or explanations."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Instructions: {prompt}\n\nFile content for {filename}:\n\n{code_content}",
        },
    ]

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {"model": chosen_model, "messages": messages, "stream": False}

    with console.status(
        f"[bold cyan]Refactoring {filename} using {chosen_model}...[/bold cyan]",
        spinner="dots",
    ):
        try:
            response = httpx.post(url_chat, json=payload, timeout=180.0)
            if response.status_code != 200:
                console.print(f"[red]Failed: {response.text}[/red]")
                raise typer.Exit(1)
            raw_text = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            console.print(f"[red]Error contacting Gateway: {e}[/red]")
            raise typer.Exit(1)

    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    # Clean markdown if present
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

    from rich.syntax import Syntax

    console.print(Syntax(cleaned_text, "python", theme="monokai", line_numbers=True))

    confirm = typer.confirm(f"\nWould you like to overwrite {filename} with these changes?")
    if confirm:
        with open(filename, "w") as f:
            f.write(cleaned_text)
        console.print(f"[bold green]Successfully updated {filename}![/bold green]")
    else:
        console.print("[yellow]Aborted.[/yellow]")


def explain_cmd(
    filename: str = typer.Argument(..., help="Path to the file to explain."),
    model_name: Optional[str] = typer.Option(None, "--model", "-m"),
):
    """Spaghetti Code Deobfuscator: Analyzes and explains the architecture and logic of a file."""
    if not os.path.exists(filename):
        console.print(f"[red]Error: File {filename} not found.[/red]")
        raise typer.Exit(1)

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        code_content = f.read()

    chosen_model = model_name if model_name else find_running_llm()
    if not chosen_model:
        console.print("[red]Error: No local LLM models found.[/red]")
        raise typer.Exit(1)

    if not auto_start_gateway():
        raise typer.Exit(1)

    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(url_load, json={"model": chosen_model}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise typer.Exit(1)

    system_prompt = (
        "You are a Senior Principal Staff Engineer. "
        "Your task is to analyze the following code file and explain it clearly and concisely.\n"
        "Provide a high-level architectural summary, break down the core entry points and data flow, "
        "and explain any obscure algorithms or design patterns used in the code.\n"
        "Use Markdown for formatting, with clear headings and bullet points."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"File content for {filename}:\n\n{code_content}"},
    ]

    url_chat = f"{get_gateway_url()}/v1/chat/completions"
    payload = {"model": chosen_model, "messages": messages, "stream": False}

    with console.status(
        f"[bold cyan]Deobfuscating {filename} using {chosen_model}...[/bold cyan]",
        spinner="dots",
    ):
        try:
            response = httpx.post(url_chat, json=payload, timeout=180.0)
            if response.status_code != 200:
                console.print(f"[red]Failed: {response.text}[/red]")
                raise typer.Exit(1)
            raw_text = response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            console.print(f"[red]Error contacting Gateway: {e}[/red]")
            raise typer.Exit(1)

    think_content, cleaned_text = extract_reasoning_and_json(raw_text)

    if think_content:
        console.print(
            Panel(
                f"[italic dim yellow]{think_content}[/italic dim yellow]",
                title="💭 Model Reasoning (CoT)",
                border_style="yellow",
                expand=False,
            )
        )

    # Strip outer markdown block if the model wrapped the entire response
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned_text = "\n".join(lines).strip()

    from rich.markdown import Markdown

    console.print(Markdown(cleaned_text))
