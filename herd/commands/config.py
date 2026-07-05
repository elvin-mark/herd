import typer
from rich.table import Table

from herd.core.config import (
    CONFIG_FILE,
    HERD_HOST,
    HERD_PORT,
    IDLE_TIMEOUT,
    LLAMA_SERVER_BIN,
    WHISPER_SERVER_BIN,
    DEFAULT_LLM,
    DEFAULT_EMBEDDING,
    DEFAULT_WHISPER,
    load_config,
    save_config,
)
from herd.core.utils import console

config_app = typer.Typer(
    name="config", help="Manage default models and environment configurations."
)


@config_app.command(name="show")
def config_show():
    """Displays all current configurations, directory paths, and default models."""
    table = Table(title="Herd Configurations")
    table.add_column("Setting / Parameter", style="cyan")
    table.add_column("Value / Model Identifier", style="magenta")
    table.add_column("Source", style="green")

    # Load file values
    disk_config = load_config()

    def add_row(key: str, active_val, desc: str):
        source = "Environment / System"
        if key in disk_config:
            source = "config.json"
        elif active_val is None:
            source = "Not Configured"
        table.add_row(key, str(active_val) if active_val is not None else "-", source)

    add_row("default_llm", DEFAULT_LLM, "Default LLM model identifier")
    add_row(
        "default_embedding", DEFAULT_EMBEDDING, "Default embedding model identifier"
    )
    add_row("default_whisper", DEFAULT_WHISPER, "Default Whisper model identifier")
    add_row("HERD_HOST", HERD_HOST, "Gateway listen host")
    add_row("HERD_PORT", HERD_PORT, "Gateway listen port")
    add_row("HERD_IDLE_TIMEOUT", IDLE_TIMEOUT, "Gateway idle timeout in seconds")
    add_row("LLAMA_SERVER_BIN", LLAMA_SERVER_BIN, "Resolved llama-server path")
    add_row("WHISPER_SERVER_BIN", WHISPER_SERVER_BIN, "Resolved whisper-server path")
    add_row("Config File Path", CONFIG_FILE, "Location of config override JSON")

    console.print("\n")
    console.print(table)
    console.print(
        "\nTo update defaults, run: [bold cyan]herd config set <key> <value>[/bold cyan]\n"
    )


@config_app.command(name="set")
def config_set(
    key: str = typer.Argument(
        ...,
        help="The setting key to modify (e.g. default_llm, default_embedding, default_whisper).",
    ),
    value: str = typer.Argument(..., help="The value to assign to the key."),
):
    """Sets a configuration value in config.json."""
    valid_keys = {
        "default_llm",
        "default_embedding",
        "default_whisper",
        "HERD_PORT",
        "HERD_HOST",
        "HERD_IDLE_TIMEOUT",
        "LLAMA_SERVER_BIN",
        "WHISPER_SERVER_BIN",
    }

    # Normalize key to lower or match
    matched_key = None
    for k in valid_keys:
        if k.lower() == key.lower():
            matched_key = k
            break

    if not matched_key:
        console.print(f"[red]Error: Invalid config key '{key}'.[/red]")
        console.print(f"Supported keys: {', '.join(sorted(list(valid_keys)))}")
        raise typer.Exit(1)

    config = load_config()

    # Type conversion if necessary
    val_to_save = value
    if matched_key in ["HERD_PORT", "HERD_IDLE_TIMEOUT"]:
        try:
            val_to_save = int(value)
        except ValueError:
            console.print(
                f"[red]Error: Value for '{matched_key}' must be an integer.[/red]"
            )
            raise typer.Exit(1)

    config[matched_key] = val_to_save
    try:
        save_config(config)
        console.print(
            f"[bold green]Success![/bold green] Configured [bold cyan]{matched_key}[/bold cyan] = [bold magenta]{val_to_save}[/bold magenta] in config.json."
        )
        console.print(
            "[yellow]Please note: Restart running gateways or processes to apply port or timeout changes.[/yellow]"
        )
    except Exception as e:
        console.print(f"[red]Failed to write configuration: {e}[/red]")
        raise typer.Exit(1)
