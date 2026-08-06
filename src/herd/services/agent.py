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

        # Path 1: Exact byte/string match (Fast Path)
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

        # Path 2: Fuzzy Normalized Line Matching (Anthropic str_replace style)
        file_lines = content.splitlines(keepends=True)
        target_lines = target.splitlines()

        norm_target = [_normalize_line(line) for line in target_lines if line.strip()]
        if not norm_target:
            return "Error: Target content to replace is empty."

        target_len = len(target_lines)
        matches = []

        # Sliding window search over file lines
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


def agent_web_search(query: str) -> str:
    """Performs zero-auth DuckDuckGo web search and returns top 5 results (title, snippet, URL)."""
    import html as html_lib
    import re
    import urllib.parse

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
            r'<a class="result__url" href="([^"]+)".*?>(.*?)</a>', raw_html, re.DOTALL
        )
        snippets = re.findall(
            r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', raw_html, re.DOTALL
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
    import html as html_lib
    import re

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
            r"<script[^>]*>.*?</script>", "", raw_html, flags=re.DOTALL | re.IGNORECASE
        )
        raw_html = re.sub(r"<style[^>]*>.*?</style>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        raw_html = re.sub(r"<nav[^>]*>.*?</nav>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
        raw_html = re.sub(
            r"<footer[^>]*>.*?</footer>", "", raw_html, flags=re.DOTALL | re.IGNORECASE
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
            if isinstance(i, str):
                try:
                    steps = json.loads(i)
                except Exception:
                    steps = [s.strip() for s in i.split("\n") if s.strip()]
            else:
                steps = i
            self.plan = [{"step": step, "status": "pending", "notes": ""} for step in steps]
            return f"Created task plan with {len(self.plan)} steps."

        def _update_plan(i):
            data = json.loads(i) if isinstance(i, str) else i
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
        if not hasattr(self, "plan") or not self.plan:
            return ""
        lines = ["\n📋 Active Task Plan Progress:"]
        for idx, item in enumerate(self.plan, 1):
            st = item["status"]
            icon = (
                "✓"
                if st == "completed"
                else ("⏳" if st == "in_progress" else ("❌" if st == "failed" else " "))
            )
            notes = f" ({item['notes']})" if item.get("notes") else ""
            lines.append(f"[{icon}] Step {idx}: {item['step']}{notes}")
        return "\n".join(lines) + "\n"

    def _compress_history_if_needed(self, max_turn_messages: int = 10, max_char_budget: int = 8000):
        """Compresses older turn history into a single structured summary block

        if message count or character threshold is exceeded. Keeps system prompt
        (index 0), objective (index 1), and recent turns intact.
        """
        if len(self.history) <= 4:
            return  # Not enough messages to compress

        turn_messages = self.history[2:]
        total_chars = sum(len(m.get("content", "")) for m in turn_messages)

        if len(turn_messages) <= max_turn_messages and total_chars <= max_char_budget:
            return

        keep_recent_count = 4
        if len(turn_messages) <= keep_recent_count:
            return

        to_compress = turn_messages[:-keep_recent_count]
        recent_to_keep = turn_messages[-keep_recent_count:]

        summary_lines = ["📜 Compressed Prior Conversation History & Progress Summary:"]

        for msg in to_compress:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "assistant":
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            summary_lines.append(f"- Agent Thought: {item.get('thought', '')}")
                            summary_lines.append(
                                f"  Action Executed: {item.get('action', '')} -> {item.get('action_input', '')}"
                            )
                    elif isinstance(data, dict):
                        summary_lines.append(f"- Agent Thought: {data.get('thought', '')}")
                        summary_lines.append(
                            f"  Action Executed: {data.get('action', '')} -> {data.get('action_input', '')}"
                        )
                except Exception:
                    summary_lines.append(f"- Assistant Action Output: {content[:150]}...")
            elif role == "user":
                if "Observation:" in content:
                    obs_text = content.replace("Observation:", "").strip()
                    if len(obs_text) > 200:
                        obs_text = obs_text[:200] + "..."
                    summary_lines.append(f"  Observation Result: {obs_text}")

        summary_block = "\n".join(summary_lines)

        self.history = [
            self.history[0],  # System prompt
            self.history[1],  # Objective
            {"role": "user", "content": summary_block},
            *recent_to_keep,
        ]

        from rich.panel import Panel

        console.print(
            Panel(
                f"[italic cyan]Compressed {len(to_compress)} older history messages into context summary block ({total_chars} chars -> {len(summary_block)} chars).[/italic cyan]",
                title="🧠 Sliding Context Window Compaction",
                border_style="blue",
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

            # Compress older turn history if context budget is exceeded
            self._compress_history_if_needed()

            # Display plan progress if available
            plan_summary = self._get_plan_summary()
            if plan_summary:
                console.print(
                    Panel(
                        plan_summary.strip(),
                        title="📋 Current Execution Plan",
                        border_style="magenta",
                    )
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

            if (
                "[" in cleaned_text
                and "]" in cleaned_text
                and ("{" not in cleaned_text or cleaned_text.find("[") < cleaned_text.find("{"))
            ):
                start_arr = cleaned_text.find("[")
                end_arr = cleaned_text.rfind("]") + 1
                cleaned_text = cleaned_text[start_arr:end_arr].strip()
            elif "{" in cleaned_text and "}" in cleaned_text:
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

            # 2. Parse action JSON (supports single action dict or batch action array)
            action_list = []
            try:
                action_data = json.loads(cleaned_text)
                if isinstance(action_data, list):
                    action_list = action_data
                elif isinstance(action_data, dict):
                    if "actions" in action_data and isinstance(action_data["actions"], list):
                        action_list = action_data["actions"]
                    else:
                        action_list = [action_data]
            except Exception:
                console.print(
                    f"[red]Error: Model output was not valid JSON. Raw output was:[/red]\n{raw_text}"
                )
                self.history.append({"role": "assistant", "content": raw_text})
                self.history.append(
                    {
                        "role": "user",
                        "content": (
                            "Please output strictly a valid JSON object or JSON array containing 'thought', 'action', and 'action_input'.\n"
                            'Example single: {"thought": "...", "action": "...", "action_input": "..."}\n'
                            'Example batch: [{"thought": "...", "action": "...", "action_input": "..."}]'
                        ),
                    }
                )
                continue

            # 3. Process actions (supporting parallel batch execution)
            observations = []
            is_final = False
            final_summary = ""

            for act_idx, act_item in enumerate(action_list, 1):
                thought = act_item.get("thought", "Executing action...")
                action = act_item.get("action", "")
                action_input = act_item.get("action_input", "")

                console.print(
                    Panel(
                        f"[italic white]{thought}[/italic white]",
                        title=f"🧠 Agent Thought (Turn {turn} - Step {act_idx}/{len(action_list)})",
                        border_style="cyan",
                    )
                )

                if action == "final_answer":
                    is_final = True
                    final_summary = str(action_input)
                    break

                console.print(
                    f"⚙️  [bold]Action:[/bold] {action} | [bold]Input:[/bold] {action_input}"
                )

                obs = self.registry.execute(action, action_input)
                console.print(
                    f"👁️  [bold]Observation:[/bold] {obs[:300]}..."
                    if len(obs) > 300
                    else f"👁️  [bold]Observation:[/bold] {obs}"
                )

                if len(action_list) > 1:
                    observations.append(f"[Action {act_idx}: {action}]\nObservation: {obs}")
                else:
                    observations.append(obs)

            if is_final:
                from rich.console import Group
                from rich.markdown import Markdown
                from rich.text import Text

                content_group = Group(
                    Text("Final Answer:", style="bold green"),
                    Markdown(final_summary),
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

            combined_obs = "\n\n".join(observations)
            plan_block = self._get_plan_summary()

            self.history.append({"role": "assistant", "content": raw_text})
            self.history.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation:\n{combined_obs}\n"
                        f"{plan_block}\n"
                        "Note: If the objective is now fully accomplished, you MUST call 'final_answer' next. "
                        "Do not repeat the same action again."
                    ),
                }
            )
