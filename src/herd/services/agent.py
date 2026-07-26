import json
import os
import subprocess
from typing import Any, Callable, Dict

import httpx

from herd.core.utils import console


class Tool:
    def __init__(self, name: str, description: str, func: Callable[[Any], str]):
        self.name = name
        self.description = description
        self.func = func


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_prompt_string(self) -> str:
        lines = ["Available Tools:"]
        for i, tool in enumerate(self.tools.values(), 1):
            lines.append(f"{i}. {tool.name}: {tool.description}")
        return "\n".join(lines) + "\n\n"

    def get_action_schema(self) -> str:
        names = ", ".join(self.tools.keys())
        return f'"The tool name to call ({names})"'

    def execute(self, action: str, action_input: Any) -> str:
        if action not in self.tools:
            return f"Error: Unknown action '{action}'."
        try:
            return self.tools[action].func(action_input)
        except Exception as e:
            return f"Error executing {action}: {e}"


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
            return (
                f"Error: start_line {start_line} exceeds total lines ({total_lines}) in '{path}'."
            )

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
    def __init__(
        self,
        model_name: str,
        gateway_url: str,
        yolo: bool = False,
        use_memory: bool = True,
    ):
        self.model_name = model_name
        self.gateway_url = gateway_url
        self.yolo = yolo
        self.use_memory = use_memory

        self.registry = ToolRegistry()
        self._register_default_tools()

        self.system_prompt = (
            "You are an autonomous AI engineering agent executing tasks in a local workspace.\n"
            "You operate in a loop: Thought -> Action -> Observation -> Repeat.\n"
            "Your goal is to satisfy the user's objective.\n\n"
        )
        self.system_prompt += self.registry.get_prompt_string()

        self.system_prompt += (
            "CRITICAL: Once the user's objective has been successfully met, you MUST immediately call the 'final_answer' tool to exit the loop.\n"
            "Do NOT perform duplicate, redundant, or repeating actions (e.g. writing the same file repeatedly) once the task is already completed.\n\n"
            "At each turn, you MUST output a valid JSON object matching the following structure:\n"
            "{\n"
            '  "thought": "What you are planning to do and why",\n'
        )

        self.system_prompt += f'  "action": {self.registry.get_action_schema()},\n'

        self.system_prompt += (
            '  "action_input": "The raw parameter string or JSON payload required by the tool"\n'
            "}\n\n"
            "Remember:\n"
            "- Do not explain your response outside of the JSON object.\n"
            "- Output strictly valid JSON. Do not wrap in markdown code blocks."
        )

        # Load Long-Term Memory
        if self.use_memory:
            memory_path = os.path.expanduser("~/.herd/agent_memory.json")
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, "r") as f:
                        memories = json.load(f)
                    if memories:
                        memory_block = "\n\nRetrieved Long-Term Memories (Context):\n"
                        for i, mem in enumerate(memories[-10:], 1):  # Max 10 recent memories
                            memory_block += f"- {mem}\n"
                        self.system_prompt += memory_block
                except Exception:
                    pass

        # Load Workspace Agent Rules
        if os.path.exists("AGENTS.md"):
            try:
                with open("AGENTS.md", "r", errors="ignore") as f:
                    agents_rules = f.read().strip()
                if agents_rules:
                    self.system_prompt += (
                        f"\n\nProject Coding Guidelines (from AGENTS.md):\n{agents_rules}\n"
                    )
            except Exception as e:
                console.print(f"[dim yellow]Warning: Could not read AGENTS.md: {e}[/dim yellow]")
        self.history = [{"role": "system", "content": self.system_prompt}]

    def _register_default_tools(self):
        self.registry.register(
            Tool(
                "list_dir",
                "List files in a folder. Action Input should be the folder path (e.g. '.' or './src').",
                lambda i: agent_list_dir(i or "."),
            )
        )
        self.registry.register(
            Tool(
                "read_file",
                "Read a text file. Action Input should be the path to the file.",
                agent_read_file,
            )
        )

        def _view_lines(i):
            data = json.loads(i) if isinstance(i, str) else i
            return agent_view_file_lines(
                data["path"], int(data["start_line"]), int(data["end_line"])
            )

        self.registry.register(
            Tool(
                "view_file_lines",
                'Read specific lines from a file. Action Input should be a JSON object containing "path", "start_line" (1-indexed, integer), and "end_line" (1-indexed, integer).',
                _view_lines,
            )
        )

        def _write_file(i):
            data = json.loads(i) if isinstance(i, str) else i
            return agent_write_file(data["path"], data["content"])

        self.registry.register(
            Tool(
                "write_file",
                'Write/overwrite a file. Action Input should be a JSON object containing "path" and "content".',
                _write_file,
            )
        )

        def _edit_file(i):
            data = json.loads(i) if isinstance(i, str) else i
            return agent_edit_file(data["path"], data["target"], data["replacement"])

        self.registry.register(
            Tool(
                "edit_file",
                'Edit a text file by replacing a unique block of target content with new content. Action Input should be a JSON object containing "path", "target", and "replacement".',
                _edit_file,
            )
        )

        def _run_cmd(i):
            if self.yolo:
                return agent_run_command(i)
            console.print(
                "\n[bold yellow]🔔 Action Approval Required:[/bold yellow] Agent wants to run shell command:"
            )
            console.print(f"  [cyan]{i}[/cyan]\n")
            import typer

            if typer.confirm("Allow execution?"):
                return agent_run_command(i)
            return "Error: User rejected execution of this command."

        self.registry.register(
            Tool(
                "run_command",
                "Run a shell command. Action Input should be the command string.",
                _run_cmd,
            )
        )

        def _search_grep(i):
            if isinstance(i, str):
                try:
                    data = json.loads(i)
                except Exception:
                    data = {"pattern": i}
            else:
                data = i
            return agent_search_grep(data.get("pattern", ""), data.get("path", "."))

        self.registry.register(
            Tool(
                "search_grep",
                'Search file contents recursively in a folder for a text pattern. Action Input should be a JSON object containing "pattern" and optionally "path" (defaults to \'.\').',
                _search_grep,
            )
        )

        def _save_memory(i):
            memory_path = os.path.expanduser("~/.herd/agent_memory.json")
            os.makedirs(os.path.dirname(memory_path), exist_ok=True)
            memories = []
            if os.path.exists(memory_path):
                with open(memory_path, "r") as f:
                    memories = json.load(f)
            memories.append(i)
            with open(memory_path, "w") as f:
                json.dump(memories, f, indent=2)
            return f"Successfully saved to long-term memory: {i}"

        if self.use_memory:
            self.registry.register(
                Tool(
                    "save_memory",
                    "Save an important project rule, context, or lesson learned to your long-term memory database. Action Input should be the text to remember.",
                    _save_memory,
                )
            )

        self.registry.register(
            Tool(
                "final_answer",
                "Signal that you have finished the objective. Action Input should be a summary of the result.",
                lambda i: "",
            )
        )

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
                from rich.console import Group
                from rich.markdown import Markdown
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

            console.print(f"⚙️  [bold]Action:[/bold] {action} | [bold]Input:[/bold] {action_input}")

            observation = self.registry.execute(action, action_input)

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
