import os
import httpx
import typer
import asyncio
import json
from typing import Optional
from rich.table import Table

from herd.core.config import (
    DEFAULT_EMBEDDING,
)
from herd.core.utils import (
    console,
    get_gateway_url,
    auto_start_gateway,
    get_local_models_info,
    find_running_llm,
)
from herd.services.rag import (
    index_directory,
    get_embedding,
    search_vectors,
    list_indexed_files,
    remove_indexed_path,
)

# Typer app for db subcommands
db_app = typer.Typer(name="db", help="Manage the local RAG vector database index.")


def index(
    directory: str = typer.Argument(..., help="Path to the local directory to index."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The embedding model identifier to use. If not specified, uses default_embedding config.",
    ),
    types: Optional[str] = typer.Option(
        None,
        "--types",
        "-t",
        help="Comma-separated file extensions to index (e.g., .py,.md).",
    ),
):
    """Recursively chunks and embeds files in a directory, storing them in the local database."""
    if not os.path.exists(directory):
        console.print(f"[red]Error: Directory does not exist: {directory}[/red]")
        raise typer.Exit(1)

    # 1. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    chosen_model = model_name if model_name else DEFAULT_EMBEDDING
    if not chosen_model:
        models = get_local_models_info()
        emb_models = [
            m["name"]
            for m in models
            if "embedding" in m["name"].lower() or "bert" in m["name"].lower()
        ]
        if emb_models:
            chosen_model = emb_models[0]

    if not chosen_model:
        console.print(
            "[red]Error: No embedding model specified and no default embedding model configured.[/red]"
        )
        raise typer.Exit(1)

    model_name = chosen_model

    # 2. Pre-load the embedding model
    console.print(
        f"Ensuring embedding model [bold magenta]{model_name}[/bold magenta] is loaded..."
    )
    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(
            url_load, json={"model": model_name, "is_embedding": True}, timeout=45.0
        )
    except Exception as e:
        console.print(f"[red]Failed to load embedding model: {e}[/red]")
        raise typer.Exit(1)

    # 3. Perform indexing
    console.print(f"Indexing directory [bold cyan]{directory}[/bold cyan]...")
    try:
        count = asyncio.run(index_directory(directory, model_name, types=types))
        console.print(
            f"[bold green]Success![/bold green] Indexed {count} text chunks in the database."
        )
    except Exception as e:
        console.print(f"[red]Failed to index directory: {e}[/red]")
        raise typer.Exit(1)


def ask(
    query: str = typer.Argument(
        ..., help="The question to ask the model using indexed context."
    ),
    model_name: Optional[str] = typer.Argument(
        None,
        help="LLM model identifier to ask. If omitted, uses active or default LLM.",
    ),
    embedding_model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The embedding model identifier to query context from. If omitted, uses default_embedding config.",
    ),
):
    """Semantic query: retrieves relevant indexed chunks and answers your question using the LLM."""
    # 1. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # Resolve LLM model
    chosen_llm = model_name if model_name else find_running_llm()
    if not chosen_llm:
        console.print(
            "[red]Error: No LLM model specified and no default LLM configured.[/red]"
        )
        raise typer.Exit(1)

    # Resolve embedding model
    chosen_emb = embedding_model if embedding_model else DEFAULT_EMBEDDING
    if not chosen_emb:
        from herd.services.rag import detect_db_embedding_model

        chosen_emb = detect_db_embedding_model()

    if not chosen_emb:
        models = get_local_models_info()
        emb_models = [
            m["name"]
            for m in models
            if "embedding" in m["name"].lower() or "bert" in m["name"].lower()
        ]
        if emb_models:
            chosen_emb = emb_models[0]

    if not chosen_emb:
        console.print(
            "[red]Error: No embedding model specified and no default embedding model configured.[/red]"
        )
        raise typer.Exit(1)

    # 2. Pre-load the models
    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(
            url_load, json={"model": chosen_emb, "is_embedding": True}, timeout=45.0
        )
        httpx.post(url_load, json={"model": chosen_llm}, timeout=45.0)
    except Exception as e:
        console.print(f"[red]Failed to load models: {e}[/red]")
        raise typer.Exit(1)

    # 3. Retrieve context
    console.print("Searching semantic index for context...")
    try:
        query_vector = asyncio.run(get_embedding(query, chosen_emb))
        matches = search_vectors(query_vector, chosen_emb, top_k=5)
    except Exception as e:
        console.print(f"[red]Failed to query embeddings: {e}[/red]")
        raise typer.Exit(1)

    if not matches:
        console.print(
            "[yellow]Warning: No context found in index for this embedding model. Answering without custom context.[/yellow]"
        )
        context = ""
    else:
        console.print("\n[bold white]Retrieved Context Sources:[/bold white]")
        for idx, m in enumerate(matches):
            basename = os.path.basename(m["file_path"])
            console.print(
                f"  [{idx + 1}] {basename} (similarity: {m['similarity']:.3f})"
            )
        context = "\n\n".join(
            [
                f"Source: {os.path.basename(m['file_path'])}\nContent:\n{m['text']}"
                for m in matches
            ]
        )

    # 4. Prompt construction
    system_prompt = (
        "You are a helpful assistant. Use the following retrieved context to answer the user's question. "
        "If you do not know the answer, say so.\n\n"
        f"Context:\n{context}"
    )

    # 5. Stream response
    async def ask_async():
        url_chat = f"{get_gateway_url()}/v1/chat/completions"
        payload = {
            "model": chosen_llm,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "stream": True,
        }
        console.print("\n[bold green]Answer:[/bold green]")
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url_chat, json=payload) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            content = data["choices"][0]["delta"].get("content", "")
                            print(content, end="", flush=True)
                        except Exception:
                            pass
        print("\n")

    try:
        asyncio.run(ask_async())
    except KeyboardInterrupt:
        console.print("\n[yellow]Generation interrupted.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")


@db_app.command(name="list")
def db_list():
    """Lists all files and paths currently indexed in the vector database."""
    try:
        rows = list_indexed_files()
    except Exception as e:
        console.print(f"[red]Error reading database: {e}[/red]")
        raise typer.Exit(1)

    if not rows:
        console.print("[yellow]The vector database index is empty.[/yellow]")
        return

    table = Table(title="Indexed Documents & Chunks")
    table.add_column("File / Directory Path", style="cyan")
    table.add_column("Embedding Model", style="magenta")
    table.add_column("Total Chunks", style="green", justify="right")

    for file_path, model_name, count in rows:
        table.add_row(file_path, model_name, str(count))

    console.print("\n")
    console.print(table)
    console.print("\n")


@db_app.command(name="search")
def db_search(
    query: str = typer.Argument(..., help="The search term to query."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="The embedding model used to query context. If omitted, uses default_embedding config.",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-l",
        help="Maximum matches to return.",
    ),
):
    """Semantic search: queries the vector database for text segments matching the query."""
    if not auto_start_gateway():
        raise typer.Exit(1)

    chosen_model = model_name if model_name else DEFAULT_EMBEDDING
    if not chosen_model:
        from herd.services.rag import detect_db_embedding_model

        chosen_model = detect_db_embedding_model()

    if not chosen_model:
        models = get_local_models_info()
        emb_models = [
            m["name"]
            for m in models
            if "embedding" in m["name"].lower() or "bert" in m["name"].lower()
        ]
        if emb_models:
            chosen_model = emb_models[0]

    if not chosen_model:
        console.print(
            "[red]Error: No embedding model specified and no default embedding model configured.[/red]"
        )
        raise typer.Exit(1)

    model_name = chosen_model

    # Pre-load the embedding model
    url_load = f"{get_gateway_url()}/v1/models/load"
    try:
        httpx.post(
            url_load, json={"model": model_name, "is_embedding": True}, timeout=45.0
        )
    except Exception as e:
        console.print(f"[red]Failed to load embedding model: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"Searching semantic index for '{query}'...")
    try:
        query_vector = asyncio.run(get_embedding(query, model_name))
        matches = search_vectors(query_vector, model_name, top_k=limit)
    except Exception as e:
        console.print(f"[red]Failed to perform semantic search: {e}[/red]")
        raise typer.Exit(1)

    if not matches:
        console.print("[yellow]No semantic matches found.[/yellow]")
        return

    table = Table(title=f"Semantic Search Matches for '{query}'")
    table.add_column("Similarity", style="green")
    table.add_column("Source File", style="cyan")
    table.add_column("Text Preview (Snippet)", style="white")

    for m in matches:
        filename = os.path.basename(m["file_path"])
        preview = m["text"][:80].replace("\n", " ") + "..."
        table.add_row(f"{m['similarity']:.3f}", filename, preview)

    console.print("\n")
    console.print(table)
    console.print("\n")


@db_app.command(name="remove")
def db_remove(
    path: str = typer.Argument(
        ..., help="The file or directory path to remove from the index."
    ),
):
    """Removes indexed chunks and files from the vector database."""
    # Resolve absolute path to match DB entries
    abs_path = os.path.abspath(path)

    console.print(
        f"Removing indexed path [bold red]{abs_path}[/bold red] from database..."
    )
    try:
        count = remove_indexed_path(abs_path)
        if count > 0:
            console.print(
                f"[bold green]Success![/bold green] Removed {count} chunks from the vector database."
            )
        else:
            console.print("[yellow]No indexed files found matching this path.[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to remove from database: {e}[/red]")
        raise typer.Exit(1)


@db_app.command(name="prune")
def db_prune():
    """Removes all indexed document chunks for files that no longer exist on the local filesystem."""
    try:
        rows = list_indexed_files()
    except Exception as e:
        console.print(f"[red]Error reading database: {e}[/red]")
        raise typer.Exit(1)

    if not rows:
        console.print("[green]The database is empty. Nothing to prune.[/green]")
        return

    pruned_files = 0
    pruned_chunks = 0

    with console.status("[bold yellow]Pruning database entries...[/bold yellow]"):
        for file_path, model_name, count in rows:
            if not os.path.exists(file_path):
                try:
                    deleted = remove_indexed_path(file_path)
                    pruned_files += 1
                    pruned_chunks += deleted
                except Exception as e:
                    console.print(f"[red]Failed to remove {file_path}: {e}[/red]")

    if pruned_files > 0:
        console.print(
            f"[bold green]Success![/bold green] Pruned [bold white]{pruned_files}[/bold white] files "
            f"([bold white]{pruned_chunks}[/bold white] chunks) that no longer exist on disk."
        )
    else:
        console.print(
            "[green]All indexed files exist on disk. No pruning needed.[/green]"
        )
