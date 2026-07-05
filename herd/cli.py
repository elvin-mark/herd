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

# Register model management commands
app.command(name="list")(list_models)
app.command(name="pull")(pull)
app.command(name="stop")(stop)
app.command(name="ps")(ps)
app.command(name="stats")(show_stats)
app.command(name="clean")(clean)
app.command(name="search")(search)
app.command(name="quantize")(quantize)
app.command(name="top")(top)

# Register chat & benchmark commands
app.command(name="run")(run)
app.command(name="benchmark")(benchmark)

# Register audio transcribing command
app.command(name="transcribe")(transcribe)

# Register developer commands
app.command(name="suggest")(suggest)
app.command(name="copilot")(copilot)
app.command(name="commit")(commit)
app.command(name="review")(review)
app.command(name="heal")(heal)
app.command(name="watch")(watch)
app.command(name="agent")(agent)

# Register RAG indexing & ask commands
app.command(name="index")(index)
app.command(name="ask")(ask)

# Register server gateway & proxy commands
app.command(name="serve")(serve)
app.command(name="logs")(logs)
app.command(name="setup")(setup)
app.command(name="share")(share)
app.command(name="proxy")(proxy)

# Register sub-Typer applications
app.add_typer(db_app)
app.add_typer(config_app)


def main():
    app()


if __name__ == "__main__":
    main()
