import os
import re

import httpx
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from herd.core.config import HERD_MODELS_DIR


def parse_model_identifier(model_name: str):
    """
    Parses a model identifier into (author, repo, tag).
    Examples:
      - 'unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M' -> ('unsloth', 'Qwen3.5-0.8B-GGUF', 'Q4_K_M')
      - 'unsloth/Qwen3.5-0.8B-GGUF' -> ('unsloth', 'Qwen3.5-0.8B-GGUF', None)
    """
    match = re.match(r"^([^/]+)/([^/:]+)(?::(.+))?$", model_name)
    if not match:
        raise ValueError(
            f"Invalid model identifier format: '{model_name}'. Expected 'author/repo[:tag]' or absolute path."
        )
    return match.groups()


def resolve_model_path(model_name: str) -> str:
    """
    Resolves a model name to a local file path.
    Supports absolute/relative paths and Hugging Face repository style.
    """
    # 1. Check if absolute or relative path
    if os.path.isabs(model_name) or model_name.startswith("./") or model_name.startswith("../"):
        if os.path.exists(model_name):
            return os.path.abspath(model_name)
        raise FileNotFoundError(f"Local model path not found: {model_name}")

    # 2. HF repository format
    author, repo, tag = parse_model_identifier(model_name)
    repo_dir = os.path.join(HERD_MODELS_DIR, "huggingface", author, repo)

    if not os.path.exists(repo_dir):
        raise FileNotFoundError(
            f"Model repository directory does not exist: {repo_dir}. Please run 'herd pull {model_name}' first."
        )

    # Get all .gguf or .bin files in repository directory
    files = [f for f in os.listdir(repo_dir) if os.path.isfile(os.path.join(repo_dir, f))]
    model_files = [f for f in files if f.endswith(".gguf") or f.endswith(".bin")]

    if not model_files:
        raise FileNotFoundError(
            f"No GGUF or BIN model files found in repository directory: {repo_dir}"
        )

    if tag:
        # Search for files matching the tag (case insensitive) but excluding mmproj first
        tagged_files = [
            f for f in model_files if tag.lower() in f.lower() and "mmproj" not in f.lower()
        ]
        if not tagged_files:
            # Fallback to any file matching the tag (including mmproj if that is specifically what's requested)
            tagged_files = [f for f in model_files if tag.lower() in f.lower()]

        if not tagged_files:
            raise FileNotFoundError(
                f"No files matching tag '{tag}' found in {repo_dir}. Available files: {model_files}"
            )
        return os.path.join(repo_dir, tagged_files[0])
    else:
        # No tag specified: try to find preferred standard quantizations first (avoid mmproj if possible)
        preferred = ["q4_k_m", "q4_0", "q8_0", "f16", "bf16"]
        for p in preferred:
            for f in model_files:
                if p in f.lower() and "mmproj" not in f.lower():
                    return os.path.join(repo_dir, f)

        # Fallback to the first non-mmproj file, or just the first file
        non_mmproj = [f for f in model_files if "mmproj" not in f.lower()]
        if non_mmproj:
            return os.path.join(repo_dir, non_mmproj[0])
        return os.path.join(repo_dir, model_files[0])


async def list_hf_repository_files(author: str, repo: str) -> list[str]:
    """Queries Hugging Face model API to list all files in the repository."""
    api_url = f"https://huggingface.co/api/models/{author}/{repo}"
    async with httpx.AsyncClient() as client:
        response = await client.get(api_url)
        if response.status_code == 404:
            raise ValueError(f"Hugging Face repository '{author}/{repo}' not found.")
        response.raise_for_status()
        data = response.json()

    siblings = data.get("siblings", [])
    files = [s["rfilename"] for s in siblings if "rfilename" in s]
    return files


async def download_file(url: str, dest_path: str, filename: str):
    """Downloads a file with a beautiful progress bar, writing to a temp file first."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_dest = dest_path + ".tmp"

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    )

    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, follow_redirects=True, timeout=None) as response:
            if response.status_code == 404:
                raise FileNotFoundError(f"File not found at URL: {url}")
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            task_id = progress.add_task(f"Downloading {filename}", total=total_size)

            with progress:
                with open(temp_dest, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))

    # Atomically move the temp file to the final destination
    os.replace(temp_dest, dest_path)
