import typer

# Import sub-commands from modular command files
from herd.commands.models import (
    list_models,
    pull,
    rm,
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
from herd.commands.server import serve, logs, setup, share, doctor

app = typer.Typer(
    name="herd",
    help="Herd - A CLI and API gateway for running local LLM and speech-to-text models (similar to Ollama).",
    no_args_is_help=True,
)

# 1. Register Core Interfaces commands
app.command(name="run", rich_help_panel="Core Interfaces")(run)
app.command(name="transcribe", rich_help_panel="Core Interfaces")(transcribe)
app.command(name="watch", rich_help_panel="Core Interfaces")(watch)

# 2. Register Model Management commands
app.command(name="list", rich_help_panel="Model Management")(list_models)
app.command(name="pull", rich_help_panel="Model Management")(pull)
app.command(name="rm", rich_help_panel="Model Management")(rm)
app.command(name="search", rich_help_panel="Model Management")(search)
app.command(name="quantize", rich_help_panel="Model Management")(quantize)
app.command(name="suggest", rich_help_panel="Model Management")(suggest)

# 3. Register Runtime Operations commands
app.command(name="ps", rich_help_panel="Runtime Operations")(ps)
app.command(name="stop", rich_help_panel="Runtime Operations")(stop)
app.command(name="top", rich_help_panel="Runtime Operations")(top)
app.command(name="stats", rich_help_panel="Runtime Operations")(show_stats)
app.command(name="logs", rich_help_panel="Runtime Operations")(logs)

# 4. Register Developer Workflows commands
app.command(name="copilot", rich_help_panel="Developer Workflows")(copilot)
app.command(name="commit", rich_help_panel="Developer Workflows")(commit)
app.command(name="review", rich_help_panel="Developer Workflows")(review)
app.command(name="heal", rich_help_panel="Developer Workflows")(heal)
app.command(name="agent", rich_help_panel="Developer Workflows")(agent)
app.command(name="benchmark", rich_help_panel="Developer Workflows")(benchmark)

# 5. Register Semantic RAG Database commands
app.command(name="index", rich_help_panel="Semantic RAG Database")(index)
app.command(name="ask", rich_help_panel="Semantic RAG Database")(ask)

# 6. Register Gateway & Configuration commands
app.command(name="serve", rich_help_panel="Gateway & Configuration")(serve)
app.command(name="setup", rich_help_panel="Gateway & Configuration")(setup)
app.command(name="share", rich_help_panel="Gateway & Configuration")(share)
app.command(name="doctor", rich_help_panel="Gateway & Configuration")(doctor)
app.command(name="clean", rich_help_panel="Gateway & Configuration")(clean)

# 7. Register Sub-Typer Applications with Rich Help Panels
app.add_typer(db_app, rich_help_panel="Semantic RAG Database")
app.add_typer(config_app, rich_help_panel="Gateway & Configuration")


def main():
    app()


if __name__ == "__main__":
    main()
