# 🤖 AI Agent Coding Guidelines & Project Conventions (AGENTS.md)

This document outlines the architectural paradigms, CLI organization schemas, testing protocols, and development workflows established for this repository. Follow these guidelines strictly when refactoring code, adding features, or implementing tests.

---

## 🏛️ 1. Architecture & Design Principles
*   **Modular APIRouters**: Keep the main web server entrypoint (e.g., `server.py`) lightweight. Modularize endpoints by feature (e.g., `routers/chat.py`, `routers/models.py`, `routers/db.py`) and keep configurations/state/exceptions separated.
*   **Path Resolution & Projector Safety**: When resolving local Hugging Face caching models by tag, prioritize files that do *not* contain companion strings like `mmproj` (projector files) to avoid mapping mismatches, falling back to all-matching tags only when explicitly requested.
*   **Idempotent Resource Pulling**: Prioritize checking local caches before initiating remote downloads to conserve bandwidth. Merge local configs with existing overrides instead of overwriting config files.

---

## 🐚 2. CLI Command Clustering (Typer Panels)
Subcommands must be logically grouped inside the `--help` menu using Typer's `rich_help_panel` parameter to maintain visual hierarchy and structure. Group subcommands into the following panels:
1.  **Core Interfaces**: Interactive, modal user-facing engines (e.g., text chat `run`, audio speech `transcribe`, vision multimodal `watch`).
2.  **Model Management**: Downloading, listing, searching, and compressing catalog files (e.g., `list`, `pull`, `search`, `quantize`, `suggest`).
3.  **Runtime Operations**: Live process listings, unloading, performance metrics, and console tail outputs (e.g., `ps`, `stop`, `top`, `stats`, `logs`).
4.  **Developer Workflows**: Code quality auditors, automations, and benchmark loops (e.g., `copilot`, `commit`, `review`, `heal`, `agent`, `benchmark`).
5.  **Semantic RAG Database**: Document chunking, indexing, and vector similarity operations.
6.  **Gateway & Configuration**: Server bindings, tunnel sharing, and global parameter overrides.

---

## 🧪 3. Non-Interactive Shell Integration Testing
To test command-line applications reliably in non-TTY (CI/CD) environments, organize integration scripts under `tests/scripts/` using these conventions:

### A. Testing Infinite / TUI Loops (`timeout` checks)
For commands that spin up blocking real-time monitors (like `herd top`) or reverse proxy servers (like `herd proxy`):
*   **Do not** pipe stdout to log files and grep for text (as interactive consoles behave differently when `stdout.isatty()` is `False`).
*   **Do** run the command inside a sandboxed `timeout` wrapper and verify that it terminates with the standard timeout exit code `124`. Any exit code other than `124` indicates a startup crash.
*   *Example:*
    ```bash
    timeout 3 herd top > /dev/null 2>&1
    if [ $? -eq 124 ]; then
        echo "Success"
    else
        echo "Crashed"
    fi
    ```

### B. Testing Interactive Prompts (`EOFError` checks)
For commands that start interactive REPL sessions (like `herd run`):
*   Piping a text prompt to stdin will trigger the first prompt execution.
*   Ensure that hitting `EOF` on the subsequent prompt raises an `EOFError` that exits the loop cleanly with status code `0`, making the subcommand natively scriptable.
*   *Example:*
    ```bash
    echo "Query text" | herd run <model>
    ```

### C. Testing Asynchronous Process Shutdowns (`sleep` checks)
When testing asynchronous shutdowns (such as unloading models with `herd stop`):
*   Insert a short `sleep` grace period (e.g., `sleep 2`) right after the stop request is made.
*   This allows the process manager backend thread to complete subprocess termination and update the active process registry before running subsequent registry state checks (like `herd ps`).

---

## 🧹 4. Linting & Formatting Compliance
Before submitting commits, run the validation toolchain:
1.  **Compilation Check**: Compile Python files to prevent syntax regressions.
    ```bash
    python3 -m py_compile **/*.py
    ```
2.  **Formatting Check**: Enforce unified code aesthetics.
    ```bash
    ruff format .
    ```
3.  **Lint Check**: Audit code quality.
    ```bash
    ruff check
    ```
