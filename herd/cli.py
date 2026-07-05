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
from herd.commands.server import serve, logs, setup, share, proxy

app = typer.Typer(
    name="herd",
    help="Herd - A CLI and API gateway for running local LLM and speech-to-text models (similar to Ollama).",
    no_args_is_help=True,
)

# 1. Define sub-Typer applications
model_app = typer.Typer(
    name="model", help="Manage local and active model server instances."
)
dev_app = typer.Typer(
    name="dev", help="Developer productivity and code-healing utilities."
)
server_app = typer.Typer(
    name="server",
    help="Manage central API gateway processes and exposing links.",
)

# 2. Register visible commands on sub-Typers
model_app.command(name="list")(list_models)
model_app.command(name="pull")(pull)
model_app.command(name="stop")(stop)
model_app.command(name="ps")(ps)
model_app.command(name="stats")(show_stats)
model_app.command(name="clean")(clean)
model_app.command(name="search")(search)
model_app.command(name="quantize")(quantize)
model_app.command(name="top")(top)

dev_app.command(name="suggest")(suggest)
dev_app.command(name="copilot")(copilot)
dev_app.command(name="commit")(commit)
dev_app.command(name="review")(review)
dev_app.command(name="heal")(heal)
dev_app.command(name="watch")(watch)
dev_app.command(name="agent")(agent)

server_app.command(name="serve")(serve)
server_app.command(name="logs")(logs)
server_app.command(name="setup")(setup)
server_app.command(name="share")(share)
server_app.command(name="proxy")(proxy)

# 3. Register root-level visible core commands
app.command(name="run")(run)
app.command(name="transcribe")(transcribe)
app.command(name="benchmark")(benchmark)

# 4. Register root-level HIDDEN alias commands (for backward compatibility & fast typing)
app.command(name="list", hidden=True)(list_models)
app.command(name="pull", hidden=True)(pull)
app.command(name="stop", hidden=True)(stop)
app.command(name="ps", hidden=True)(ps)
app.command(name="stats", hidden=True)(show_stats)
app.command(name="clean", hidden=True)(clean)
app.command(name="search", hidden=True)(search)
app.command(name="quantize", hidden=True)(quantize)
app.command(name="top", hidden=True)(top)

app.command(name="suggest", hidden=True)(suggest)
app.command(name="copilot", hidden=True)(copilot)
app.command(name="commit", hidden=True)(commit)
app.command(name="review", hidden=True)(review)
app.command(name="heal", hidden=True)(heal)
app.command(name="watch", hidden=True)(watch)
app.command(name="agent", hidden=True)(agent)

app.command(name="index", hidden=True)(index)
app.command(name="ask", hidden=True)(ask)

app.command(name="serve", hidden=True)(serve)
app.command(name="logs", hidden=True)(logs)
app.command(name="setup", hidden=True)(setup)
app.command(name="share", hidden=True)(share)
app.command(name="proxy", hidden=True)(proxy)

# 5. Register all sub-Typer applications
app.add_typer(model_app)
app.add_typer(dev_app)
app.add_typer(server_app)
app.add_typer(db_app)
app.add_typer(config_app)


def main():
    app()


if __name__ == "__main__":
    main()
