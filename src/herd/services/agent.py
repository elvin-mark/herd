"""
Herd Autonomous AI Engineering Agent
Refactored with modular tool handlers, decoupled LLM transport,
robust JSON action parsers, and UI event listeners.
"""

import html as html_lib
import json
import os
import re
import subprocess
import urllib.parse
from typing import Any, Callable, Dict, List, Optional

import httpx

from herd.core.utils import console

# -----------------------------------------------------------------------------
# 1. Event Listener Interface (UI Decoupling)
# -----------------------------------------------------------------------------


class AgentEventListener:
    """Decouples CLI console rendering from core agent logic."""

    def on_turn_start(self, turn: int, max_turns: int):
        console.print(f"\n[bold cyan]── Agent Iteration {turn}/{max_turns} ──[/bold cyan]")

    def on_plan_update(self, plan_summary: str):
        console.print(f"[dim]{plan_summary}[/dim]")

    def on_thought(self, thought: str):
        console.print(f"🤔 [bold white]Thought:[/bold white] {thought}")

    def on_action(self, action: str, action_input: Any):
        console.print(
            f"🛠️  [bold yellow]Action:[/bold yellow] {action} -> [cyan]{action_input}[/cyan]"
        )

    def on_observation(self, observation: str):
        snippet = observation if len(observation) <= 400 else observation[:400] + "... [truncated]"
        console.print(f"👀 [bold green]Observation:[/bold green]\n{snippet}\n")

    def on_compaction(self, msg_count: int, orig_chars: int, comp_chars: int):
        console.print(
            f"\n[bold blue]🧠 Sliding Context Window Compaction[/bold blue]\n"
            f"[dim]Compressed {msg_count} older history messages ({orig_chars} -> {comp_chars} chars).[/dim]\n"
        )

    def on_finish(self, answer: str):
        console.print(f"\n🎯 [bold green]Final Answer:[/bold green]\n{answer}\n")

    def on_error(self, error_msg: str):
        console.print(f"[bold red]Agent Error:[/bold red] {error_msg}")


# -----------------------------------------------------------------------------
# 2. Decoupled LLM Gateway Client
# -----------------------------------------------------------------------------


class LLMClient:
    """Transport provider interface for communicating with gateway models."""

    def __init__(self, gateway_url: str, model_name: str):
        self.gateway_url = gateway_url.rstrip("/")
        self.model_name = model_name

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        url = f"{self.gateway_url}/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        resp = httpx.post(url, json=payload, timeout=120.0)
        if resp.status_code != 200:
            raise RuntimeError(f"Gateway HTTP {resp.status_code}: {resp.text}")

        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"]
        # Strip CoT <think>...</think> tags if present
        return re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()


# -----------------------------------------------------------------------------
# 3. Robust Action & JSON Parser
# -----------------------------------------------------------------------------


def parse_agent_action(response_text: str) -> List[Dict[str, Any]]:
    """Parses LLM output text into one or more structured action dicts."""
    text = response_text.strip()
    if not text:
        return [
            {
                "thought": "No response returned",
                "action": "final_answer",
                "action_input": "Empty response.",
            }
        ]

    # Strip markdown wrapper if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()

    # Repair common JSON trailing commas
    cleaned_json = re.sub(r",\s*([\]}])", r"\1", text)

    try:
        data = json.loads(cleaned_json)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except Exception:
        pass

    # Regex extraction fallback for {"thought": ..., "action": ...}
    match = re.search(r"\{[\s\S]*\"action\"\s*:[\s\S]*\}", text)
    if match:
        try:
            return [json.loads(match.group(0))]
        except Exception:
            pass

    return [
        {
            "thought": "Failed to parse JSON response; treating output as final answer.",
            "action": "final_answer",
            "action_input": text,
        }
    ]


# -----------------------------------------------------------------------------
# 4. Tool Registry & Standalone Helper Functions
# -----------------------------------------------------------------------------


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


def _parse_json_args(raw_input: Any) -> Any:
    """Helper to safely parse tool input arguments."""
    if isinstance(raw_input, str):
        try:
            return json.loads(raw_input)
        except Exception:
            return raw_input
    return raw_input


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
        output = [f"{idx}: {line}" for idx, line in enumerate(sliced, start=start_line)]
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


def _normalize_line(line: str) -> str:
    """Normalizes whitespace and tabs for fuzzy line matching."""
    return line.expandtabs(4).strip()


def agent_edit_file(path: str, target: str, replacement: str) -> str:
    import difflib

    try:
        if not os.path.exists(path):
            return f"Error: File '{path}' does not exist."
        with open(path, "r", errors="ignore") as f:
            content = f.read()

        # Path 1: Exact match
        if target in content:
            count = content.count(target)
            if count > 1:
                return (
                    f"Error: Target content occurs {count} times in '{path}'. "
                    "Please specify 2-3 extra surrounding lines of context to make the edit location unique."
                )
            new_content = content.replace(target, replacement)
            with open(path, "w") as f:
                f.write(new_content)
            return f"Successfully edited file '{path}' (exact match)."

        # Path 2: Fuzzy Normalized Line Matching
        file_lines = content.splitlines(keepends=True)
        target_lines = target.splitlines()

        norm_target = [_normalize_line(line) for line in target_lines if line.strip()]
        if not norm_target:
            return "Error: Target content to replace is empty."

        target_len = len(target_lines)
        matches = []

        for i in range(len(file_lines) - target_len + 1):
            window_raw = file_lines[i : i + target_len]
            norm_window = [_normalize_line(line) for line in window_raw if line.strip()]

            if norm_window == norm_target:
                matches.append((i, i + target_len, 1.0))
            else:
                matcher = difflib.SequenceMatcher(None, norm_window, norm_target)
                ratio = matcher.ratio()
                if ratio >= 0.85:
                    matches.append((i, i + target_len, ratio))

        if len(matches) == 1:
            start_idx, end_idx, confidence = matches[0]
            rep_text = (
                replacement if replacement.endswith("\n") or not file_lines else replacement + "\n"
            )
            new_lines = file_lines[:start_idx] + [rep_text] + file_lines[end_idx:]
            with open(path, "w") as f:
                f.writelines(new_lines)

            match_type = (
                "exact line match"
                if confidence == 1.0
                else f"fuzzy match ({confidence:.0%} confidence)"
            )
            return f"Successfully edited file '{path}' (applied {match_type})."

        elif len(matches) > 1:
            return (
                f"Error: Target content matches {len(matches)} potential locations in '{path}'. "
                "Please include 2-3 additional surrounding lines of code to clarify the exact location."
            )

        return (
            f"Error: Could not locate target content in '{path}'. "
            "Please view the file first using 'view_file_lines' or 'read_file' to ensure your target block includes accurate surrounding lines of context."
        )
    except Exception as e:
        return f"Error editing file: {e}"


def agent_run_command(command: str) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = proc.stdout
        if proc.stderr:
            output += "\n" + proc.stderr
        return (
            output.strip()
            if output.strip()
            else f"Command executed cleanly with returncode {proc.returncode}."
        )
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out after 120 seconds."
    except Exception as e:
        return f"Error running command: {e}"


def agent_search_grep(pattern: str, path: str = ".") -> str:
    matches = []
    ignored_dirs = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".herd",
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


def agent_web_search(query: str) -> str:
    """Performs zero-auth DuckDuckGo web search and returns top 5 results (title, snippet, URL)."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=10.0)
        if resp.status_code != 200:
            return f"Error: Web search request failed with status code {resp.status_code}."

        raw_html = resp.text
        results = []
        links = re.findall(
            r'<a class="result__url" href="([^"]+)".*?>(.*?)</a>',
            raw_html,
            re.DOTALL,
        )
        snippets = re.findall(
            r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>',
            raw_html,
            re.DOTALL,
        )
        titles = re.findall(r'<a class="result__a"[^>]*>(.*?)</a>', raw_html, re.DOTALL)

        for idx in range(min(5, len(titles))):
            t_raw = titles[idx] if idx < len(titles) else "Result"
            s_raw = snippets[idx] if idx < len(snippets) else ""
            u_raw = links[idx][0] if idx < len(links) else ""

            t_clean = html_lib.unescape(re.sub(r"<[^>]+>", "", t_raw)).strip()
            s_clean = html_lib.unescape(re.sub(r"<[^>]+>", "", s_raw)).strip()
            u_clean = re.sub(r"<[^>]+>", "", u_raw).strip()

            if u_clean.startswith("//"):
                u_clean = "https:" + u_clean

            results.append(f"{idx + 1}. [{t_clean}]({u_clean})\n   Snippet: {s_clean}")

        if not results:
            return f"No web search results found for query: '{query}'."

        return f"🔍 Web Search Results for '{query}':\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Error performing web search: {e}"


def agent_fetch_url(url: str) -> str:
    """Fetches web page content, strips HTML chrome, and returns clean readable text."""
    try:
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "https://" + url

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        resp = httpx.get(url, headers=headers, follow_redirects=True, timeout=12.0)
        if resp.status_code != 200:
            return f"Error: Failed to fetch URL '{url}' (status code {resp.status_code})."

        raw_html = resp.text
        raw_html = re.sub(
            r"<script[^>]*>.*?</script>",
            "",
            raw_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        raw_html = re.sub(
            r"<style[^>]*>.*?</style>",
            "",
            raw_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        raw_html = re.sub(
            r"<nav[^>]*>.*?</nav>",
            "",
            raw_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        raw_html = re.sub(
            r"<footer[^>]*>.*?</footer>",
            "",
            raw_html,
            flags=re.DOTALL | re.IGNORECASE,
        )

        raw_html = re.sub(r"<(h[1-6]|p|div|li|tr)[^>]*>", "\n", raw_html, flags=re.IGNORECASE)
        raw_html = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)

        text = re.sub(r"<[^>]+>", " ", raw_html)
        text = html_lib.unescape(text)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)

        if len(cleaned_text) > 3500:
            cleaned_text = cleaned_text[:3500] + "\n... [Content truncated to 3500 characters]"

        return f"🌐 Content from '{url}':\n\n{cleaned_text}"
    except Exception as e:
        return f"Error fetching URL '{url}': {e}"


# -----------------------------------------------------------------------------
# 5. Agent Session Manager
# -----------------------------------------------------------------------------


class AgentSession:
    def __init__(
        self,
        model_name: str,
        gateway_url: str,
        yolo: bool = False,
        use_memory: bool = True,
        listener: Optional[AgentEventListener] = None,
    ):
        self.model_name = model_name
        self.gateway_url = gateway_url
        self.yolo = yolo
        self.use_memory = use_memory
        self.listener = listener or AgentEventListener()

        self.client = LLMClient(gateway_url, model_name)
        self.registry = ToolRegistry()
        self.plan: List[Dict[str, Any]] = []

        self._register_default_tools()

        self.system_prompt = (
            "You are an autonomous AI engineering agent executing tasks in a local workspace.\n"
            "You operate in a loop: Thought -> Action -> Observation -> Repeat.\n"
            "Your goal is to satisfy the user's objective.\n\n"
        )
        self.system_prompt += self.registry.get_prompt_string()
        self.system_prompt += (
            "CRITICAL: Once the user's objective has been successfully met, you MUST immediately call the 'final_answer' tool to exit the loop.\n"
            "Do NOT perform duplicate, redundant, or repeating actions once the task is already completed.\n\n"
            "At each turn, you MUST output a valid JSON object matching the following structure:\n"
            "{\n"
            '  "thought": "What you are planning to do and why",\n'
            f'  "action": {self.registry.get_action_schema()},\n'
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
                        for mem in memories[-10:]:
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
            data = _parse_json_args(i)
            return agent_view_file_lines(
                data["path"], int(data["start_line"]), int(data["end_line"])
            )

        self.registry.register(
            Tool(
                "view_file_lines",
                'View a specific slice of lines from a text file. Action Input should be a JSON object containing "path", "start_line", and "end_line".',
                _view_lines,
            )
        )

        def _write_file(i):
            data = _parse_json_args(i)
            if isinstance(data, dict):
                return agent_write_file(data["path"], data["content"])
            return "Error: write_file expects JSON object with 'path' and 'content'."

        self.registry.register(
            Tool(
                "write_file",
                'Write content to a file. Action Input should be a JSON object containing "path" and "content".',
                _write_file,
            )
        )

        def _edit_file(i):
            data = _parse_json_args(i)
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
            data = _parse_json_args(i)
            if isinstance(data, dict):
                return agent_search_grep(data.get("pattern", ""), data.get("path", "."))
            return agent_search_grep(str(i), ".")

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
                "web_search",
                "Perform a live web search for documentation, code examples, or library APIs. Action Input should be the search query string.",
                agent_web_search,
            )
        )

        self.registry.register(
            Tool(
                "fetch_url",
                "Fetch and read the text content of a web page URL. Action Input should be the full URL (e.g. 'https://docs.python.org/...').",
                agent_fetch_url,
            )
        )

        def _create_plan(i):
            steps = _parse_json_args(i)
            if isinstance(steps, str):
                steps = [s.strip() for s in steps.split("\n") if s.strip()]
            self.plan = [{"step": step, "status": "pending", "notes": ""} for step in steps]
            return f"Created task plan with {len(self.plan)} steps."

        def _update_plan(i):
            data = _parse_json_args(i)
            step_num = int(data.get("step_number", 1)) - 1
            if 0 <= step_num < len(self.plan):
                self.plan[step_num]["status"] = data.get("status", "completed")
                if "notes" in data:
                    self.plan[step_num]["notes"] = data["notes"]
                return f"Updated step {step_num + 1} status to '{self.plan[step_num]['status']}'."
            return f"Error: Invalid step_number {step_num + 1}. Plan has {len(self.plan)} steps."

        self.registry.register(
            Tool(
                "create_plan",
                'Break down a large/complex task into sequential steps. Action Input should be a JSON array of step strings e.g. ["Step 1: Read files", "Step 2: Implement feature", "Step 3: Run tests"].',
                _create_plan,
            )
        )

        self.registry.register(
            Tool(
                "update_plan",
                'Update status of a planned step. Action Input should be a JSON object containing "step_number" (1-indexed integer), "status" ("completed", "in_progress", "failed"), and optionally "notes".',
                _update_plan,
            )
        )

        self.registry.register(
            Tool(
                "final_answer",
                "Signal that you have finished the objective. Action Input should be a summary of the result.",
                lambda i: "",
            )
        )

    def _get_plan_summary(self) -> str:
        if not self.plan:
            return ""
        lines = ["\n📋 Active Task Plan Progress:"]
        for idx, item in enumerate(self.plan, 1):
            mark = (
                "[✓]"
                if item["status"] == "completed"
                else ("[▶]" if item["status"] == "in_progress" else "[ ]")
            )
            notes_str = f" ({item['notes']})" if item.get("notes") else ""
            lines.append(f"{mark} Step {idx}: {item['step']}{notes_str}")
        return "\n".join(lines) + "\n"

    def _compress_history_if_needed(self, max_turn_messages: int = 10, max_char_budget: int = 8000):
        turn_history = self.history[2:]
        if not turn_history:
            return

        total_chars = sum(len(m.get("content", "")) for m in turn_history)
        if len(turn_history) <= max_turn_messages and total_chars <= max_char_budget:
            return

        recent_turns = turn_history[-4:]
        older_turns = turn_history[:-4]

        if not older_turns:
            return

        summary_lines = ["📜 Compressed Prior Conversation History & Progress Summary:"]
        for msg in older_turns:
            role = msg.get("role", "")
            content = msg.get("content", "").strip()

            if role == "assistant":
                actions = parse_agent_action(content)
                for act in actions:
                    thought = act.get("thought", "")
                    action = act.get("action", "")
                    action_in = act.get("action_input", "")
                    if thought:
                        summary_lines.append(f"- Agent Thought: {thought}")
                    if action:
                        summary_lines.append(f"  Action Executed: {action} -> {action_in}")
            elif role == "user" and content.startswith("Observation:"):
                obs_snippet = content[12:].strip()
                if len(obs_snippet) > 150:
                    obs_snippet = obs_snippet[:150] + "..."
                summary_lines.append(f"  Observation Result: {obs_snippet}")

        summary_text = "\n".join(summary_lines)
        self.listener.on_compaction(len(older_turns), total_chars, len(summary_text))

        self.history = [
            self.history[0],
            self.history[1],
            {"role": "user", "content": summary_text},
            *recent_turns,
        ]

    def run_task(self, objective: str, max_turns: int = 10) -> Optional[str]:
        self.history.append({"role": "user", "content": f"Objective: {objective}"})

        for turn in range(1, max_turns + 1):
            self._compress_history_if_needed()
            self.listener.on_turn_start(turn, max_turns)

            plan_str = self._get_plan_summary()
            if plan_str:
                self.listener.on_plan_update(plan_str)

            try:
                response_text = self.client.chat_completion(self.history)
            except Exception as e:
                self.listener.on_error(str(e))
                return f"Error connecting to model server: {e}"

            self.history.append({"role": "assistant", "content": response_text})

            actions = parse_agent_action(response_text)
            observations = []
            is_finished = False

            for act in actions:
                thought = act.get("thought", "No thought provided.")
                action = act.get("action")
                action_input = act.get("action_input")

                self.listener.on_thought(thought)
                if action:
                    self.listener.on_action(action, action_input)

                if not action:
                    observations.append("Error: Invalid JSON response; missing 'action' field.")
                    continue

                if action == "final_answer":
                    final_res = str(action_input)
                    self.listener.on_finish(final_res)
                    is_finished = True
                    break

                observation = self.registry.execute(action, action_input)
                self.listener.on_observation(observation)
                observations.append(f"Action '{action}' Result:\n{observation}")

            if is_finished:
                return self.history[-1]["content"]

            combined_obs = "\n\n".join(observations)
            self.history.append({"role": "user", "content": f"Observation:\n{combined_obs}"})

        self.listener.on_error(f"Agent reached maximum execution turns limit ({max_turns}).")
        return "Task incomplete: Reached maximum turn iterations."
