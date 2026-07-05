import typer

# Import sub-commands from modular command files
from herd.commands.models import (
    list_models,
    pull,
    stop,
    ps,
    show_stats,
    clean,
    search,
    quantize,
    top,
)
from herd.commands.chat import run, benchmark
from herd.commands.audio import transcribe
from herd.commands.developer import suggest, copilot, commit, review, heal, watch, agent
from herd.commands.db import index, ask, db_app
from herd.commands.config import config_app
from herd.commands.server import serve, logs, setup, share, proxy, doctor

app = typer.Typer(
    name="herd",
    help="Herd - A CLI and API gateway for running local LLM and speech-to-text models (similar to Ollama).",
    no_args_is_help=True,
)

# 1. Register Core Interfaces commands
app.command(name="run", rich_help_panel="Core Interfaces")(run)
app.command(name="transcribe", rich_help_panel="Core Interfaces")(transcribe)
app.command(name="benchmark", rich_help_panel="Core Interfaces")(benchmark)

# 2. Register Model Management commands
app.command(name="list", rich_help_panel="Model Management")(list_models)
app.command(name="pull", rich_help_panel="Model Management")(pull)
app.command(name="stop", rich_help_panel="Model Management")(stop)
app.command(name="ps", rich_help_panel="Model Management")(ps)
app.command(name="stats", rich_help_panel="Model Management")(show_stats)
app.command(name="clean", rich_help_panel="Model Management")(clean)
app.command(name="search", rich_help_panel="Model Management")(search)
app.command(name="quantize", rich_help_panel="Model Management")(quantize)
app.command(name="top", rich_help_panel="Model Management")(top)

# 3. Register Developer Tools commands
app.command(name="suggest", rich_help_panel="Developer Tools")(suggest)
app.command(name="copilot", rich_help_panel="Developer Tools")(copilot)
app.command(name="commit", rich_help_panel="Developer Tools")(commit)
app.command(name="review", rich_help_panel="Developer Tools")(review)
app.command(name="heal", rich_help_panel="Developer Tools")(heal)
app.command(name="watch", rich_help_panel="Developer Tools")(watch)
app.command(name="agent", rich_help_panel="Developer Tools")(agent)

# 4. Register Semantic RAG Database commands
app.command(name="index", rich_help_panel="Semantic RAG Database")(index)
app.command(name="ask", rich_help_panel="Semantic RAG Database")(ask)

# 5. Register Gateway & Configuration commands
app.command(name="serve", rich_help_panel="Gateway & Configuration")(serve)
app.command(name="logs", rich_help_panel="Gateway & Configuration")(logs)
app.command(name="setup", rich_help_panel="Gateway & Configuration")(setup)
app.command(name="share", rich_help_panel="Gateway & Configuration")(share)
app.command(name="proxy", rich_help_panel="Gateway & Configuration")(proxy)
app.command(name="doctor", rich_help_panel="Gateway & Configuration")(doctor)

# 6. Register Sub-Typer Applications with Rich Help Panels
app.add_typer(db_app, rich_help_panel="Semantic RAG Database")
app.add_typer(config_app, rich_help_panel="Gateway & Configuration")


def main():
    app()


if __name__ == "__main__":
    main()
