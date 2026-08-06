import mimetypes
import os
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from herd.api.exceptions import HerdError

router = APIRouter()


def get_workspace_dir() -> str:
    """Returns the canonical workspace directory path."""
    return os.path.realpath(os.getcwd())


def is_path_in_workspace(path: str) -> bool:
    """Verifies that target path resolves within the workspace directory boundary."""
    abs_path = os.path.realpath(os.path.abspath(path))
    workspace_path = get_workspace_dir()
    return abs_path == workspace_path or abs_path.startswith(workspace_path + os.sep)


@router.get("/v1/workspace/files")
async def list_workspace_files(subpath: Optional[str] = Query(None)):
    """Lists files and folders inside the workspace directory."""
    base_dir = get_workspace_dir()
    target_dir = base_dir

    if subpath:
        requested_path = os.path.join(base_dir, subpath)
        if not is_path_in_workspace(requested_path):
            raise HerdError(
                status_code=403,
                message="Access denied: target path is outside workspace directory boundary.",
            )
        target_dir = os.path.realpath(requested_path)

    if not os.path.exists(target_dir):
        raise HerdError(
            status_code=444 if False else 404,
            message=f"Directory not found: '{subpath or '.'}'",
        )

    ignored = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".herd",
        ".ruff_cache",
        ".pytest_cache",
        "build",
        "dist",
    }
    items = []

    try:
        for entry in os.listdir(target_dir):
            if entry in ignored or entry.startswith("."):
                continue
            full_path = os.path.join(target_dir, entry)
            rel_path = os.path.relpath(full_path, base_dir)
            is_dir = os.path.isdir(full_path)
            size = os.path.getsize(full_path) if not is_dir else 0

            # Categorize file type
            ext = os.path.splitext(entry)[1].lower()
            category = "file"
            if is_dir:
                category = "folder"
            elif ext in [".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"]:
                category = "image"
            elif ext in [".html", ".htm"]:
                category = "html"
            elif ext in [".md", ".markdown"]:
                category = "markdown"

            items.append(
                {
                    "name": entry,
                    "path": rel_path,
                    "is_dir": is_dir,
                    "size": size,
                    "category": category,
                }
            )
        return {
            "workspace": base_dir,
            "files": sorted(items, key=lambda x: (not x["is_dir"], x["name"].lower())),
        }
    except Exception as e:
        raise HerdError(status_code=500, message=f"Failed to list workspace files: {str(e)}")


@router.get("/v1/workspace/file")
async def get_workspace_file(path: str = Query(...)):
    """Serves a file from the workspace for rendering or download."""
    base_dir = get_workspace_dir()
    full_path = os.path.join(base_dir, path)

    if not is_path_in_workspace(full_path):
        raise HerdError(
            status_code=403,
            message="Security Violation: Access denied outside workspace directory.",
        )

    if not os.path.exists(full_path) or os.path.isdir(full_path):
        raise HerdError(status_code=404, message=f"File not found: '{path}'")

    mime_type, _ = mimetypes.guess_type(full_path)
    if not mime_type:
        mime_type = "application/octet-stream"

    return FileResponse(
        path=full_path,
        media_type=mime_type,
        filename=os.path.basename(full_path),
    )
