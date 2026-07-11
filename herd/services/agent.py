import os
import json
import subprocess
import httpx
from herd.core.utils import console


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


def agent_view_file_lines(path: str, start_line: int, end_line: int) -> str:
    try:
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        if start_line < 1 or end_line < start_line:
            return "Error: Invalid line range. Ensure start_line >= 1 and end_line >= start_line."

        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()

        total_lines = len(lines)
        if start_line > total_lines:
            return f"Error: start_line {start_line} exceeds total lines ({total_lines}) in '{path}'."

        sliced = lines[start_line - 1 : min(end_line, total_lines)]
        output = []
        for idx, line in enumerate(sliced, start=start_line):
            output.append(f"{idx}: {line}")

        return "".join(output)
    except Exception as e:
        return f"Error reading file lines: {e}"


def agent_write_file(path: str, content: str) -> str:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"File successfully written to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def agent_edit_file(path: str, target: str, replacement: str) -> str:
    try:
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        with open(path, "r", errors="ignore") as f:
            content = f.read()
        if target not in content:
            return f"Error: Target content to replace not found in '{path}'. Make sure it matches exactly (including leading whitespace)."
        count = content.count(target)
        if count > 1:
            return f"Error: Target content occurs {count} times in '{path}'. Please specify a more unique block of code (including surrounding lines) to edit."
        new_content = content.replace(target, replacement)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Successfully edited file '{path}' (replaced target content)."
    except Exception as e:
        return f"Error editing file: {e}"


def agent_run_command(command: str) -> str:
    try:
        res = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30.0
        )
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


def agent_search_grep(pattern: str, path: str = ".") -> str:
    matches = []
    ignored_dirs = {
        ".git",
        ".venv",
        "node_modules",
        "__pycache__",
        ".ruff_cache",
        ".pytest_cache",
        "build",
        "dist",
    }
    try:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ignored_dirs]
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", errors="ignore") as f:
                        for line_no, line in enumerate(f, 1):
                            if pattern.lower() in line.lower():
                                matches.append(f"{file_path}:{line_no}: {line.strip()}")
                                if len(matches) >= 50:
                                    break
                except Exception:
                    continue
                if len(matches) >= 50:
                    break
        if not matches:
            return f"No matches found for pattern '{pattern}'."
        return "\n".join(matches)
    except Exception as e:
        return f"Error executing search_grep: {e}"


class AgentSession:
    def __init__(self, model_name: str, gateway_url: str, yolo: bool = False):
        self.model_name = model_name
        self.gateway_url = gateway_url
        self.yolo = yolo

        self.system_prompt = (
            "You are an autonomous AI engineering agent executing tasks in a local workspace.\n"
            "You operate in a loop: Thought -> Action -> Observation -> Repeat.\n"
            "Your goal is to satisfy the user's objective.\n\n"
            "Available Tools:\n"
            "1. list_dir: List files in a folder. Action Input should be the folder path (e.g. '.' or './src').\n"
            "2. read_file: Read a text file. Action Input should be the path to the file.\n"
            '3. view_file_lines: Read specific lines from a file. Action Input should be a JSON object containing "path", "start_line" (1-indexed, integer), and "end_line" (1-indexed, integer).\n'
            '4. write_file: Write/overwrite a file. Action Input should be a JSON object containing "path" and "content".\n'
            '5. edit_file: Edit a text file by replacing a unique block of target content with new content. Action Input should be a JSON object containing "path", "target", and "replacement".\n'
            "6. run_command: Run a shell command. Action Input should be the command string.\n"
            '7. search_grep: Search file contents recursively in a folder for a text pattern. Action Input should be a JSON object containing "pattern" and optionally "path" (defaults to \'.\').\n'
            "8. final_answer: Signal that you have finished the objective. Action Input should be a summary of the result.\n\n"
            "CRITICAL: Once the user's objective has been successfully met, you MUST immediately call the 'final_answer' tool to exit the loop.\n"
            "Do NOT perform duplicate, redundant, or repeating actions (e.g. writing the same file repeatedly) once the task is already completed.\n\n"
            "At each turn, you MUST output a valid JSON object matching the following structure:\n"
            "{\n"
            '  "thought": "What you are planning to do and why",\n'
            '  "action": "The tool name to call (list_dir, read_file, view_file_lines, write_file, edit_file, run_command, search_grep, final_answer)",\n'
            '  "action_input": "The raw parameter string or JSON payload required by the tool"\n'
            "}\n\n"
            "Remember:\n"
            "- Do not explain your response outside of the JSON object.\n"
            "- Output strictly valid JSON. Do not wrap in markdown code blocks."
        )

        self.history = [{"role": "system", "content": self.system_prompt}]

    def run_task(self, objective: str, max_turns: int = 10):
        from rich.panel import Panel

        self.history.append({"role": "user", "content": f"Objective: {objective}"})

        url_chat = f"{self.gateway_url}/v1/chat/completions"

        for turn in range(1, max_turns + 1):
            console.print(
                f"[bold dim]── Turn {turn}/{max_turns} ──────────────────────────────────────[/bold dim]"
            )

            # 1. Ask LLM for next step
            try:
                res = httpx.post(
                    url_chat,
                    json={
                        "model": self.model_name,
                        "messages": self.history,
                        "temperature": 0.2,
                        "stream": False,
                    },
                    headers={"Accept-Encoding": "identity"},
                    timeout=60.0,
                )
                if res.status_code != 200:
                    console.print(f"[red]Error from Gateway: {res.text}[/red]")
                    break
                raw_text = res.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                console.print(f"[red]Error communicating with LLM: {e}[/red]")
                break

            # Extract think block if present and format JSON text
            cleaned_text = raw_text.strip()
            think_content = ""
            if "<think>" in cleaned_text and "</think>" in cleaned_text:
                start_think = cleaned_text.find("<think>") + 7
                end_think = cleaned_text.find("</think>")
                think_content = cleaned_text[start_think:end_think].strip()
                cleaned_text = cleaned_text[end_think + 8 :].strip()

            if "{" in cleaned_text and "}" in cleaned_text:
                start_json = cleaned_text.find("{")
                end_json = cleaned_text.rfind("}") + 1
                cleaned_text = cleaned_text[start_json:end_json].strip()

            if think_content:
                console.print(
                    Panel(
                        f"[italic dim yellow]{think_content}[/italic dim yellow]",
                        title="💭 Model Reasoning (CoT)",
                        border_style="yellow",
                    )
                )

            # 2. Parse action JSON
            try:
                action_data = json.loads(cleaned_text)
                thought = action_data["thought"]
                action = action_data["action"]
                action_input = action_data["action_input"]
            except Exception:
                console.print(
                    f"[red]Error: Model output was not valid JSON. Raw output was:[/red]\n{raw_text}"
                )
                self.history.append({"role": "assistant", "content": raw_text})
                self.history.append(
                    {
                        "role": "user",
                        "content": "Please output strictly a valid JSON object containing 'thought', 'action', and 'action_input'.",
                    }
                )
                continue

            # Print thought
            console.print(
                Panel(
                    f"[italic white]{thought}[/italic white]",
                    title=f"🧠 Agent Thought (Turn {turn})",
                    border_style="cyan",
                )
            )

            # 3. Handle Actions
            if action == "final_answer":
                from rich.markdown import Markdown
                from rich.console import Group
                from rich.text import Text

                content_group = Group(
                    Text("Final Answer:", style="bold green"),
                    Markdown(action_input),
                )
                console.print(
                    Panel(
                        content_group,
                        title="🏁 Objective Accomplished",
                        border_style="green",
                    )
                )
                self.history.append({"role": "assistant", "content": raw_text})
                break

            console.print(
                f"⚙️  [bold]Action:[/bold] {action} | [bold]Input:[/bold] {action_input}"
            )

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
                    observation = agent_write_file(
                        write_data["path"], write_data["content"]
                    )
                except Exception as e:
                    observation = f"Error parsing write_file parameters: {e}. Expected a JSON object with 'path' and 'content'."
            elif action == "edit_file":
                try:
                    if isinstance(action_input, str):
                        edit_data = json.loads(action_input)
                    else:
                        edit_data = action_input
                    observation = agent_edit_file(
                        edit_data["path"],
                        edit_data["target"],
                        edit_data["replacement"],
                    )
                except Exception as e:
                    observation = f"Error parsing edit_file parameters: {e}. Expected a JSON object with 'path', 'target', and 'replacement'."
            elif action == "view_file_lines":
                try:
                    if isinstance(action_input, str):
                        view_data = json.loads(action_input)
                    else:
                        view_data = action_input
                    observation = agent_view_file_lines(
                        view_data["path"],
                        int(view_data["start_line"]),
                        int(view_data["end_line"]),
                    )
                except Exception as e:
                    observation = f"Error parsing view_file_lines parameters: {e}. Expected a JSON object with 'path', 'start_line', and 'end_line'."
            elif action == "run_command":
                if self.yolo:
                    observation = agent_run_command(action_input)
                else:
                    console.print(
                        "\n[bold yellow]🔔 Action Approval Required:[/bold yellow] Agent wants to run shell command:"
                    )
                    console.print(f"  [cyan]{action_input}[/cyan]\n")
                    import typer

                    confirm = typer.confirm("Allow execution?")
                    if confirm:
                        observation = agent_run_command(action_input)
                    else:
                        observation = "Error: User rejected execution of this command."
            elif action == "search_grep":
                try:
                    if isinstance(action_input, str):
                        try:
                            search_data = json.loads(action_input)
                        except Exception:
                            search_data = {"pattern": action_input}
                    else:
                        search_data = action_input

                    pattern = search_data.get("pattern", "")
                    search_path = search_data.get("path", ".")
                    observation = agent_search_grep(pattern, search_path)
                except Exception as e:
                    observation = f"Error parsing search_grep parameters: {e}. Expected a JSON object with 'pattern' and optional 'path'."
            else:
                observation = f"Error: Unknown action '{action}'."

            # Show observation
            console.print(
                f"👁️  [bold]Observation:[/bold] {observation[:400]}..."
                if len(observation) > 400
                else f"👁️  [bold]Observation:[/bold] {observation}"
            )

            self.history.append({"role": "assistant", "content": raw_text})
            self.history.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation: {observation}\n\n"
                        "Note: If the objective is now fully accomplished, you MUST call 'final_answer' next. "
                        "Do not repeat the same action again."
                    ),
                }
            )
