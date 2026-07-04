import os
import httpx
import typer
from typing import Optional

from herd.core.config import (
    DEFAULT_WHISPER,
)
from herd.core.utils import (
    console,
    get_gateway_url,
    auto_start_gateway,
    get_local_models_info,
)


def ms_to_srt_time(ms: int) -> str:
    """Converts milliseconds to SRT time format HH:MM:SS,mmm"""
    secs, msecs = divmod(ms, 1000)
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{msecs:03d}"


def ms_to_vtt_time(ms: int) -> str:
    """Converts milliseconds to VTT time format HH:MM:SS.mmm"""
    secs, msecs = divmod(ms, 1000)
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"


def transcribe(
    audio_file: str = typer.Argument(..., help="Path to the local audio file to transcribe."),
    model_name: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Whisper model identifier. If not specified, auto-selects the first locally downloaded Whisper model.",
    ),
    output_format: str = typer.Option(
        "txt",
        "--format",
        "-f",
        help="Output transcription format (txt, srt, vtt).",
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the output transcription. Defaults to <audio_file_name>.<format>.",
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Target language code (e.g. en, es, fr). If not set, auto-detects language.",
    ),
):
    """Transcribes an audio file into text, subtitle formats (SRT/VTT), or raw text using Whisper."""
    # 1. Ensure audio file exists
    if not os.path.exists(audio_file):
        console.print(f"[red]Error: Audio file not found at: {audio_file}[/red]")
        raise typer.Exit(1)

    # 2. Ensure gateway is running
    if not auto_start_gateway():
        raise typer.Exit(1)

    # 3. Resolve Whisper model to use
    chosen_model = model_name
    if not chosen_model:
        if DEFAULT_WHISPER:
            chosen_model = DEFAULT_WHISPER
        else:
            whisper_models = [
                m["name"]
                for m in get_local_models_info()
                if "whisper" in m["name"].lower() or m["filename"].endswith(".bin")
            ]
            if whisper_models:
                chosen_model = whisper_models[0]
                console.print(f"[yellow]No model specified. Auto-selected local Whisper model: [bold]{chosen_model}[/bold][/yellow]")

    if not chosen_model:
        console.print("[red]Error: No Whisper models found locally and no default Whisper model configured.[/red]")
        console.print("Please download a Whisper model first: [bold cyan]herd pull ggerganov/whisper.cpp:ggml-base.en.bin[/bold cyan]")
        raise typer.Exit(1)

    # 4. Resolve output path
    fmt = output_format.lower()
    if fmt not in ["txt", "srt", "vtt"]:
        console.print(f"[red]Error: Unsupported output format '{output_format}'. Choose from: txt, srt, vtt.[/red]")
        raise typer.Exit(1)

    if not output_file:
        base, _ = os.path.splitext(audio_file)
        dest_path = f"{base}.{fmt}"
    else:
        dest_path = output_file

    console.print(f"Loading Whisper model [bold cyan]{chosen_model}[/bold cyan] in Gateway...")

    # 5. Send transcription request to the Gateway
    url = f"{get_gateway_url()}/v1/audio/transcriptions"

    # Open and stream file
    try:
        with open(audio_file, "rb") as f_bin:
            files = {"file": (os.path.basename(audio_file), f_bin, "audio/wav")}
            data = {
                "model": chosen_model,
                "response_format": "json"
            }
            if language:
                data["language"] = language

            console.print("[bold green]Transcribing audio file...[/bold green] (this may take a few moments)")
            response = httpx.post(url, files=files, data=data, timeout=None)
    except Exception as e:
        console.print(f"[red]Error contacting Gateway transcription server: {e}[/red]")
        raise typer.Exit(1)

    if response.status_code != 200:
        console.print(f"[red]Transcription failed: {response.text}[/red]")
        raise typer.Exit(1)

    result = response.json()

    # 6. Format and save the transcription
    segments = result.get("segments", [])

    try:
        with open(dest_path, "w", encoding="utf-8") as f:
            if fmt == "txt":
                f.write(result.get("text", "").strip())
            elif fmt == "srt":
                for idx, seg in enumerate(segments):
                    from_ms = seg.get("offsets", {}).get("from", 0)
                    to_ms = seg.get("offsets", {}).get("to", 0)
                    start_str = ms_to_srt_time(from_ms)
                    end_str = ms_to_srt_time(to_ms)
                    text = seg.get("text", "").strip()
                    f.write(f"{idx + 1}\n{start_str} --> {end_str}\n{text}\n\n")
            elif fmt == "vtt":
                f.write("WEBVTT\n\n")
                for idx, seg in enumerate(segments):
                    from_ms = seg.get("offsets", {}).get("from", 0)
                    to_ms = seg.get("offsets", {}).get("to", 0)
                    start_str = ms_to_vtt_time(from_ms)
                    end_str = ms_to_vtt_time(to_ms)
                    text = seg.get("text", "").strip()
                    f.write(f"{start_str} --> {end_str}\n{text}\n\n")

        console.print(f"\n[bold green]Success![/bold green] Transcription saved to: [bold cyan]{dest_path}[/bold cyan]")
    except Exception as e:
        console.print(f"[red]Error writing transcription file: {e}[/red]")
        raise typer.Exit(1)
