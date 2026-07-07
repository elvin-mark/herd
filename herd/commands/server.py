import os
import time
import json
import subprocess
import shutil
import typer
import uvicorn
import re
import threading
import signal
import socket
import platform
import psutil
import httpx
from collections import deque
from typing import Optional

from herd.core.config import (
    HERD_HOST,
    HERD_PORT,
    HERD_HOME,
    HERD_LOGS_DIR,
)
from herd.core.utils import console


def start_public_tunnel(port: int):
    cloudflared_bin = shutil.which("cloudflared")
    if not cloudflared_bin:
        console.print(
            "[red]Error: 'cloudflared' is not installed or not in PATH.[/red]"
        )
        console.print("Please install Cloudflare Tunnel first. Examples:")
        console.print("  [bold white]macOS:[/bold white] brew install cloudflared")
        console.print("  [bold white]Linux:[/bold white] sudo apt install cloudflared")
        return None

    console.print("[bold cyan]Starting public Cloudflare Tunnel...[/bold cyan]")
    try:
        process = subprocess.Popen(
            [cloudflared_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        return process
    except Exception as e:
        console.print(f"[red]Failed to start Cloudflare Tunnel: {e}[/red]")
        return None


def run_tunnel_monitor(process, port):
    # Read lines to find the trycloudflare URL
    public_url = None
    start_time = time.time()
    while time.time() - start_time < 15.0:  # 15s timeout
        line = process.stdout.readline()
        if not line:
            break

        match = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", line)
        if match:
            public_url = match.group(1)
            break

    if not public_url:
        console.print(
            "[red]Error: Failed to retrieve Cloudflare Tunnel URL. Check if cloudflared is working correctly.[/red]"
        )
        process.terminate()
        process.wait()
        return

    console.print("\n🌎 [bold green]Public Exposure Active![/bold green]\n")
    console.print(f"  Public API Base URL:  [bold cyan]{public_url}/v1[/bold cyan]")
    console.print(f"  Public Web Dashboard: [bold cyan]{public_url}[/bold cyan]")
    console.print("")
    console.print(
        "[yellow]Your local Herd gateway is now securely accessible from anywhere in the world![/yellow]\n"
    )


def serve(
    host: str = typer.Option(
        HERD_HOST,
        "--host",
        "-h",
        help="Host IP address to bind the gateway server to (use '0.0.0.0' for local network access).",
    ),
    port: int = typer.Option(
        HERD_PORT, "--port", "-p", help="Port to run the gateway server on."
    ),
    public: bool = typer.Option(
        False,
        "--public",
        help="Expose the gateway to the public internet using a free Cloudflare Tunnel.",
    ),
):
    """Starts the central Herd API Gateway server."""
    # Ensure gateway port and host are set in env so other processes know about it
    os.environ["HERD_PORT"] = str(port)
    os.environ["HERD_HOST"] = host
    console.print(
        f"[bold green]Starting Herd API Gateway on {host}:{port}...[/bold green]"
    )

    tunnel_proc = None
    if public:
        tunnel_proc = start_public_tunnel(port)
        if tunnel_proc:

            def monitor():
                time.sleep(2.0)
                run_tunnel_monitor(tunnel_proc, port)

            t = threading.Thread(target=monitor, daemon=True)
            t.start()

    try:
        # Correct path to the FastAPI app module under the new package layout
        uvicorn.run("herd.api.server:app", host=host, port=port, log_level="info")
    finally:
        if tunnel_proc:
            console.print("\n[yellow]Stopping Cloudflare Tunnel...[/yellow]")
            try:
                os.killpg(os.getpgid(tunnel_proc.pid), signal.SIGTERM)
                tunnel_proc.wait()
            except Exception:
                try:
                    tunnel_proc.terminate()
                    tunnel_proc.wait()
                except Exception:
                    pass
            console.print("[green]Public URL revoked successfully.[/green]")


def logs(
    model_name: Optional[str] = typer.Argument(
        None,
        help="Model identifier to view logs for. If omitted, tails the gateway logs.",
    ),
    follow: bool = typer.Option(
        False, "--follow", "-f", help="Follow log output in real-time."
    ),
    lines: int = typer.Option(
        20, "--lines", "-n", help="Number of lines to show from the end of the logs."
    ),
):
    """Views or live-tails logs for a model process or the central gateway."""
    if model_name:
        model_safe = model_name.replace("/", "_").replace(":", "_")
        log_path = os.path.join(HERD_LOGS_DIR, f"{model_safe}.log")
        target_desc = f"Model '{model_name}'"
    else:
        log_path = os.path.join(HERD_LOGS_DIR, "gateway.log")
        target_desc = "Herd Gateway"

    if not os.path.exists(log_path):
        console.print(f"[red]No logs found at: {log_path}[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold green]Tailing last {lines} lines of {target_desc} logs...[/bold green]"
    )

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            last_lines = deque(f, maxlen=lines)
            for line in last_lines:
                print(line, end="")

            if follow:
                f.seek(0, 2)
                console.print(
                    "\n[bold yellow]--- Following logs (Press Ctrl+C to exit) ---[/bold yellow]\n"
                )
                while True:
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        continue
                    print(line, end="", flush=True)
    except KeyboardInterrupt:
        console.print("\n[yellow]Log tailing stopped.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error reading logs: {e}[/red]")


def get_git_commit(repo_dir: str) -> Optional[str]:
    """Retrieves the current git commit hash of the specified repository directory."""
    try:
        git_bin = shutil.which("git")
        if not git_bin:
            return None
        res = subprocess.run(
            [git_bin, "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return None


def setup(
    dir_path: Optional[str] = typer.Option(
        None,
        "--dir",
        "-d",
        help="Directory where llama.cpp and whisper.cpp will be cloned and compiled. Defaults to HERD_HOME/src.",
    ),
    cuda: bool = typer.Option(
        False, "--cuda", help="Compile llama.cpp and whisper.cpp with CUDA support."
    ),
):
    """Clones, compiles, and configures llama.cpp and whisper.cpp locally."""
    if not dir_path:
        dir_path = os.path.join(HERD_HOME, "src")

    os.makedirs(dir_path, exist_ok=True)

    git_bin = shutil.which("git")
    cmake_bin = shutil.which("cmake")
    if not git_bin:
        console.print(
            "[red]Error: 'git' is not installed or not in PATH. Please install git first.[/red]"
        )
        raise typer.Exit(1)
    if not cmake_bin:
        console.print(
            "[red]Error: 'cmake' is not installed or not in PATH. Please install cmake first.[/red]"
        )
        raise typer.Exit(1)

    llama_dir = os.path.join(dir_path, "llama.cpp")
    whisper_dir = os.path.join(dir_path, "whisper.cpp")

    # 1. Setup llama.cpp
    if not os.path.exists(llama_dir):
        console.print("[bold cyan]Cloning llama.cpp...[/bold cyan]")
        subprocess.run(
            [
                git_bin,
                "clone",
                "--depth",
                "1",
                "https://github.com/ggerganov/llama.cpp.git",
                llama_dir,
            ],
            check=True,
        )
    else:
        console.print(
            "[yellow]llama.cpp directory already exists. Skipping clone.[/yellow]"
        )

    console.print("[bold cyan]Compiling llama-server...[/bold cyan]")
    cmake_args = [cmake_bin, "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
    if cuda:
        cmake_args.append("-DGGML_CUDA=ON")

    cores = os.cpu_count() or 1
    subprocess.run(cmake_args, cwd=llama_dir, check=True)
    subprocess.run(
        [
            cmake_bin,
            "--build",
            "build",
            "--config",
            "Release",
            "--target",
            "llama-server",
            "--parallel",
            str(cores),
        ],
        cwd=llama_dir,
        check=True,
    )

    # 2. Setup whisper.cpp
    if not os.path.exists(whisper_dir):
        console.print("[bold cyan]Cloning whisper.cpp...[/bold cyan]")
        subprocess.run(
            [
                git_bin,
                "clone",
                "--depth",
                "1",
                "https://github.com/ggerganov/whisper.cpp.git",
                whisper_dir,
            ],
            check=True,
        )
    else:
        console.print(
            "[yellow]whisper.cpp directory already exists. Skipping clone.[/yellow]"
        )

    console.print("[bold cyan]Compiling whisper-server...[/bold cyan]")
    whisper_cmake_args = [cmake_bin, "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
    if cuda:
        whisper_cmake_args.append("-DGGML_CUDA=ON")

    subprocess.run(whisper_cmake_args, cwd=whisper_dir, check=True)
    subprocess.run(
        [
            cmake_bin,
            "--build",
            "build",
            "--config",
            "Release",
            "--target",
            "whisper-server",
            "--parallel",
            str(cores),
        ],
        cwd=whisper_dir,
        check=True,
    )

    # 3. Configure binary paths
    llama_bin_path = os.path.abspath(
        os.path.join(llama_dir, "build", "bin", "llama-server")
    )
    whisper_bin_path = os.path.abspath(
        os.path.join(whisper_dir, "build", "bin", "whisper-server")
    )
    if not os.path.exists(whisper_bin_path):
        fallback_path = os.path.abspath(
            os.path.join(whisper_dir, "build", "whisper-server")
        )
        if os.path.exists(fallback_path):
            whisper_bin_path = fallback_path

    llama_commit = get_git_commit(llama_dir)
    whisper_commit = get_git_commit(whisper_dir)

    config_path = os.path.join(HERD_HOME, "config.json")
    from herd.core.config import load_config, save_config

    config_data = load_config()
    config_data["LLAMA_SERVER_BIN"] = llama_bin_path
    config_data["WHISPER_SERVER_BIN"] = whisper_bin_path
    if llama_commit:
        config_data["LLAMA_COMMIT"] = llama_commit
    if whisper_commit:
        config_data["WHISPER_COMMIT"] = whisper_commit

    save_config(config_data)

    console.print("\n[bold green]Herd setup completed successfully![/bold green]")
    console.print(
        f"Custom binary paths registered in [bold cyan]{config_path}[/bold cyan]:"
    )
    console.print(f"  llama-server:   [bold white]{llama_bin_path}[/bold white]")
    if llama_commit:
        console.print(f"    (Commit: [bold white]{llama_commit}[/bold white])")
    console.print(f"  whisper-server: [bold white]{whisper_bin_path}[/bold white]")
    if whisper_commit:
        console.print(f"    (Commit: [bold white]{whisper_commit}[/bold white])")


def get_local_ip() -> str:
    """Finds the primary local IP address of this machine."""

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def share(
    qr: bool = typer.Option(
        False,
        "--qr",
        "-q",
        help="Generate an ASCII QR code in the terminal for easy mobile pairing.",
    ),
    public: bool = typer.Option(
        False,
        "--public",
        "-p",
        help="Expose the gateway to the public internet using a free Cloudflare Tunnel.",
    ),
):
    """Exposes connection strings and generates pairing helper for local network or public devices."""
    port = HERD_PORT

    if public:
        cloudflared_bin = shutil.which("cloudflared")
        if not cloudflared_bin:
            console.print(
                "[red]Error: 'cloudflared' is not installed or not in PATH.[/red]"
            )
            console.print("Please install Cloudflare Tunnel first. Examples:")
            console.print("  [bold white]macOS:[/bold white] brew install cloudflared")
            console.print(
                "  [bold white]Linux:[/bold white] sudo apt install cloudflared"
            )
            raise typer.Exit(1)

        console.print("[bold cyan]Starting public Cloudflare Tunnel...[/bold cyan]")
        try:
            process = subprocess.Popen(
                [cloudflared_bin, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            # Read lines to find the trycloudflare URL
            public_url = None
            start_time = time.time()
            while time.time() - start_time < 15.0:  # 15s timeout
                line = process.stdout.readline()
                if not line:
                    break

                match = re.search(r"(https://[a-zA-Z0-9-]+\.trycloudflare\.com)", line)
                if match:
                    public_url = match.group(1)
                    break

            if not public_url:
                console.print(
                    "[red]Error: Failed to retrieve Cloudflare Tunnel URL. Check if cloudflared is working correctly.[/red]"
                )
                process.terminate()
                process.wait()
                raise typer.Exit(1)

            console.print("\n🌎 [bold green]Public Exposure Active![/bold green]\n")
            console.print(
                f"  Public API Base URL:  [bold cyan]{public_url}/v1[/bold cyan]"
            )
            console.print(
                f"  Public Web Dashboard: [bold cyan]{public_url}[/bold cyan]"
            )
            console.print("")
            console.print(
                "[yellow]Your local Herd gateway is now securely accessible from anywhere in the world![/yellow]"
            )

            if qr:
                try:
                    import qrcode

                    console.print(
                        "\n[bold yellow]Scan this QR Code to copy the Public API URL on your mobile device:[/bold yellow]\n"
                    )
                    qr_obj = qrcode.QRCode()
                    qr_obj.add_data(f"{public_url}/v1")
                    qr_obj.make()
                    qr_obj.print_ascii(tty=True)
                    console.print("")
                except ImportError:
                    pass

            console.print(
                "[bold yellow]--- Press Ctrl+C to stop the tunnel and revoke the public URL ---[/bold yellow]\n"
            )

            # Block and keep reading to keep process alive, print errors if any
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                # Silently consume output to avoid terminal clutter, but keep loop alive
                pass

        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping Cloudflare Tunnel...[/yellow]")
        finally:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait()
            except Exception:
                try:
                    process.terminate()
                    process.wait()
                except Exception:
                    pass
            console.print("[green]Public URL revoked successfully.[/green]")
        return

    # Default local share logic
    ip = get_local_ip()
    url = f"http://{ip}:{port}/v1"

    console.print("\n📶 [bold green]Herd Connection & Exposer Helper[/bold green]\n")
    console.print("Your Gateway is accessible on the local network at:")
    console.print(f"  API Base URL:  [bold cyan]{url}[/bold cyan]")
    console.print(f"  Web Dashboard: [bold cyan]http://{ip}:{port}[/bold cyan]")
    console.print("")
    console.print(
        "Configure your mobile client (e.g. Chatbox, LibreChat) with this API Base URL."
    )
    console.print("")

    if qr:
        try:
            import qrcode

            console.print(
                "[bold yellow]Scan this QR Code to copy the API Base URL on your mobile device:[/bold yellow]\n"
            )
            qr_obj = qrcode.QRCode()
            qr_obj.add_data(url)
            qr_obj.make()
            qr_obj.print_ascii(tty=True)
            console.print("")
        except ImportError:
            console.print(
                "[yellow]Notice: 'qrcode' package is not installed. To display QR codes, install it via:[/yellow]"
            )
            console.print("  [bold cyan]pip install qrcode[/bold cyan]")
            console.print("")


def proxy(
    remote_url: str = typer.Argument(
        ...,
        help="The remote Herd gateway URL to proxy requests to (e.g. http://192.168.1.100:11434).",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Host interface to bind the local proxy gateway to.",
    ),
    port: int = typer.Option(
        HERD_PORT, "--port", "-p", help="Port to run the local proxy gateway on."
    ),
):
    """Starts a local reverse proxy that forwards all API requests transparently to a remote Herd instance."""
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
    import httpx

    proxy_app = FastAPI(title="Herd Gateway Proxy")
    target_base = remote_url.rstrip("/")

    console.print(
        f"Starting Herd Proxy Gateway on [bold cyan]{host}:{port}[/bold cyan] -> [bold magenta]{target_base}[/bold magenta]..."
    )

    @proxy_app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    )
    async def reverse_proxy_route(request: Request, path: str):
        remote_url_str = f"{target_base}/{path}"
        query = request.url.query
        if query:
            remote_url_str = f"{remote_url_str}?{query}"

        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ["host", "content-length"]
        }

        async def stream_generator():
            async with httpx.AsyncClient(timeout=None) as client:
                try:
                    async with client.stream(
                        method=request.method,
                        url=remote_url_str,
                        headers=headers,
                        content=body,
                    ) as response:
                        async for chunk in response.aiter_bytes():
                            yield chunk
                except Exception as e:
                    yield json.dumps({"error": f"Proxy request failed: {e}"}).encode()

        return StreamingResponse(stream_generator(), media_type="application/json")

    try:
        uvicorn.run(proxy_app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping proxy gateway...[/yellow]")


def doctor():
    """Audits system environment, CPU instruction sets, GPU capabilities, and local server status."""

    console.print("\n[bold cyan]📋 Herd System Doctor Diagnosis[/bold cyan]\n")

    # 1. Host OS Info
    console.print("[bold yellow]1. Operating System Details:[/bold yellow]")
    console.print(f"  OS Type:      [bold white]{platform.system()}[/bold white]")
    console.print(f"  Kernel:       [bold white]{platform.release()}[/bold white]")
    console.print(f"  Architecture: [bold white]{platform.machine()}[/bold white]")

    # 2. CPU Capabilities
    console.print("\n[bold yellow]2. Processor Capabilities:[/bold yellow]")
    cores_physical = psutil.cpu_count(logical=False)
    cores_logical = psutil.cpu_count(logical=True)
    console.print(
        f"  Cores:        [bold white]{cores_physical} physical, {cores_logical} logical[/bold white]"
    )

    # Try to scan for AVX flags on Linux/MacOS
    cpu_flags = []
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.strip().startswith("flags"):
                        flags = line.split(":", 1)[1].strip().split()
                        for flag in ["avx", "avx2", "avx512f", "fma"]:
                            if flag in flags:
                                cpu_flags.append(flag.upper())
                        break
        elif platform.system() == "Darwin":
            res = subprocess.run(["sysctl", "-a"], capture_output=True, text=True)
            for flag in ["AVX1_0", "AVX2", "AVX512F", "FMA"]:
                if flag in res.stdout:
                    cpu_flags.append(flag.replace("1_0", "").upper())
    except Exception:
        pass

    if cpu_flags:
        console.print(
            f"  CPU Flags:    [bold green]{', '.join(cpu_flags)}[/bold green] (Inference-capable)"
        )
    else:
        console.print(
            "  CPU Flags:    [bold white]Standard instruction set[/bold white]"
        )

    # 3. GPU/CUDA Capabilities
    console.print("\n[bold yellow]3. GPU / Hardware Acceleration:[/bold yellow]")
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            res = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if res.returncode == 0:
                gpu_lines = res.stdout.strip().split("\n")
                for line in gpu_lines:
                    parts = line.split(",")
                    name = parts[0].strip()
                    total = parts[1].strip()
                    free = parts[2].strip()
                    console.print(f"  GPU Device:   [bold green]{name}[/bold green]")
                    console.print(
                        f"  VRAM:         [bold green]{free} MB free / {total} MB total[/bold green]"
                    )
            else:
                console.print(
                    "  GPU Device:   [bold yellow]NVIDIA Driver present but query failed[/bold yellow]"
                )
        except Exception:
            console.print(
                "  GPU Device:   [bold yellow]Error querying nvidia-smi[/bold yellow]"
            )
    else:
        console.print(
            "  GPU Device:   [bold white]No NVIDIA GPU detected (CPU mode active)[/bold white]"
        )

    # 4. Binary Dependencies
    console.print("\n[bold yellow]4. Compiled Server Binaries:[/bold yellow]")
    config_path = os.path.join(HERD_HOME, "config.json")
    llama_bin = None
    whisper_bin = None
    llama_commit = None
    whisper_commit = None
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                llama_bin = cfg.get("LLAMA_SERVER_BIN")
                whisper_bin = cfg.get("WHISPER_SERVER_BIN")
                llama_commit = cfg.get("LLAMA_COMMIT")
                whisper_commit = cfg.get("WHISPER_COMMIT")
        except Exception:
            pass

    # Check llama-server
    if not llama_bin or not os.path.exists(llama_bin):
        llama_bin = shutil.which("llama-server")
    if llama_bin and os.path.exists(llama_bin):
        commit_str = f" | Commit: {llama_commit}" if llama_commit else ""
        console.print(
            f"  llama-server:   [bold green]Found[/bold green] ({llama_bin}){commit_str}"
        )
    else:
        console.print(
            "  llama-server:   [bold red]Missing[/bold red] (Run 'herd setup' to compile)"
        )

    # Check whisper-server
    if not whisper_bin or not os.path.exists(whisper_bin):
        whisper_bin = shutil.which("whisper-server")
    if whisper_bin and os.path.exists(whisper_bin):
        commit_str = f" | Commit: {whisper_commit}" if whisper_commit else ""
        console.print(
            f"  whisper-server: [bold green]Found[/bold green] ({whisper_bin}){commit_str}"
        )
    else:
        console.print(
            "  whisper-server: [bold red]Missing[/bold red] (Run 'herd setup' to compile)"
        )

    # 5. Gateway Server Status
    console.print("\n[bold yellow]5. Herd Gateway Server Link:[/bold yellow]")
    gateway_url = f"http://{HERD_HOST}:{HERD_PORT}"
    try:
        res = httpx.get(f"{gateway_url}/health", timeout=1.0)
        if res.status_code == 200:
            console.print(
                f"  Connection:   [bold green]Online[/bold green] ({gateway_url})"
            )
        else:
            console.print(
                f"  Connection:   [bold red]Offline[/bold red] (Status code {res.status_code})"
            )
    except Exception:
        console.print(
            f"  Connection:   [bold red]Offline[/bold red] (Gateway not running on port {HERD_PORT})"
        )
    console.print("")
