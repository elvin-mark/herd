import typer

from herd.commands.audio import transcribe
from herd.commands.chat import benchmark, run
from herd.commands.config import config_app
from herd.commands.db import ask, db_app, index
from herd.commands.developer import (
    agent,
    commit,
    copilot,
    docs_cmd,
    explain_cmd,
    heal,
    pr,
    refactor_cmd,
    review,
    test_cmd,
    triage,
    vision,
)

# Import sub-commands from modular command files
from herd.commands.models import (
    clean,
    history_cmd,
    list_models,
    pool_app,
    ps,
    pull,
    rm,
    search,
    show_stats,
    stop,
    top,
)
from herd.commands.server import doctor, logs, serve, setup, share

app = typer.Typer(
    name="herd",
    help="Herd - A CLI and API gateway for running local LLM and speech-to-text models (similar to Ollama).",
    no_args_is_help=True,
)

# 1. Register Core Interfaces commands
app.command(name="run", rich_help_panel="Core Interfaces")(run)
app.command(name="transcribe", rich_help_panel="Core Interfaces")(transcribe)
app.command(name="vision", rich_help_panel="Core Interfaces")(vision)

# 2. Register Model Management commands
app.command(name="list", rich_help_panel="Model Management")(list_models)
app.command(name="pull", rich_help_panel="Model Management")(pull)
app.command(name="rm", rich_help_panel="Model Management")(rm)
app.command(name="search", rich_help_panel="Model Management")(search)
app.add_typer(pool_app, rich_help_panel="Model Management")


# 3. Register Runtime Operations commands
app.command(name="ps", rich_help_panel="Runtime Operations")(ps)
app.command(name="stop", rich_help_panel="Runtime Operations")(stop)
app.command(name="top", rich_help_panel="Runtime Operations")(top)
app.command(name="stats", rich_help_panel="Runtime Operations")(show_stats)
app.command(name="logs", rich_help_panel="Runtime Operations")(logs)
app.command(name="history", rich_help_panel="Runtime Operations")(history_cmd)


# 4. Register Developer Workflows commands
dev_app = typer.Typer(name="dev", help="Developer productivity tools and generators.")
dev_app.command(name="copilot")(copilot)
dev_app.command(name="heal")(heal)
dev_app.command(name="agent")(agent)
dev_app.command(name="benchmark")(benchmark)
dev_app.command(name="test")(test_cmd)
dev_app.command(name="docs")(docs_cmd)
dev_app.command(name="refactor")(refactor_cmd)
dev_app.command(name="explain")(explain_cmd)
app.add_typer(dev_app, rich_help_panel="Tools")

# 5. Bind RAG core commands to the db subgroup
db_app.command(name="index")(index)
db_app.command(name="ask")(ask)

# 6. Register Gateway & Configuration commands
app.command(name="serve", rich_help_panel="Gateway & Configuration")(serve)
app.command(name="setup", rich_help_panel="Gateway & Configuration")(setup)
app.command(name="share", rich_help_panel="Gateway & Configuration")(share)
app.command(name="doctor", rich_help_panel="Gateway & Configuration")(doctor)
app.command(name="clean", rich_help_panel="Gateway & Configuration")(clean)

# 7. Register Sub-Typer Applications with Rich Help Panels
app.add_typer(db_app, rich_help_panel="Tools")
app.add_typer(config_app, rich_help_panel="Gateway & Configuration")

git_app = typer.Typer(name="git", help="Git automations for committing and code review.")
git_app.command(name="commit")(commit)
git_app.command(name="review")(review)
git_app.command(name="triage")(triage)
git_app.command(name="pr")(pr)
app.add_typer(git_app, rich_help_panel="Tools")


def main():
    app()


if __name__ == "__main__":
    main()
