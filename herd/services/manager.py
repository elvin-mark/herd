import os
import time
import asyncio
import logging
import socket
from typing import Dict, Any, Optional
import httpx
from herd.core.config import (
    HERD_LOGS_DIR,
    LLAMA_SERVER_BIN,
    WHISPER_SERVER_BIN,
    IDLE_TIMEOUT,
)
from herd.services.downloader import resolve_model_path

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger("herd.manager")


def find_free_port() -> int:
    """Finds an available TCP port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ProcessManager:
    def __init__(self):
        self.running_models: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

    async def get_or_start_server(
        self,
        model_name: str,
        is_whisper: bool = False,
        is_embedding: bool = False,
        idle_timeout: Optional[int] = None,
    ) -> int:
        """
        Retrieves the port of a running model server, starting it if not already running.
        Updates the last accessed time of the server.
        """
        async with self.lock:
            # 1. Resolve model file path
            try:
                model_path = resolve_model_path(model_name)
            except Exception as e:
                logger.error(f"Error resolving model path for '{model_name}': {e}")
                raise

            # 2. Check if model is already running by its file path
            if model_path in self.running_models:
                info = self.running_models[model_path]
                info["last_accessed"] = time.time()

                # Verify that the process is still running
                if info["process"].returncode is None:
                    # Check if the startup flags match
                    if (
                        info["is_whisper"] == is_whisper
                        and info["is_embedding"] == is_embedding
                    ):
                        if idle_timeout is not None:
                            info["idle_timeout"] = idle_timeout
                        return info["port"]
                    else:
                        logger.info(
                            f"Model '{model_name}' running flags mismatched (whisper={info['is_whisper']}->{is_whisper}, "
                            f"embedding={info['is_embedding']}->{is_embedding}). Restarting server with new flags..."
                        )
                        # Stop the mismatched process inline
                        process = info["process"]
                        log_file = info["log_file"]
                        try:
                            process.terminate()
                            try:
                                await asyncio.wait_for(process.wait(), timeout=5.0)
                            except asyncio.TimeoutError:
                                logger.warning(
                                    f"Process for '{model_name}' did not exit. Force killing..."
                                )
                                process.kill()
                                await process.wait()
                        except Exception as e:
                            logger.error(
                                f"Error terminating model server for restart: {e}"
                            )
                        finally:
                            try:
                                log_file.close()
                            except Exception:
                                pass
                        self.running_models.pop(model_path)

            # 3. Find a free port
            port = find_free_port()

            # 4. Construct command line
            if is_whisper:
                cmd = [
                    WHISPER_SERVER_BIN,
                    "--model",
                    model_path,
                    "--port",
                    str(port),
                    "--host",
                    "127.0.0.1",
                ]
            else:
                cmd = [
                    LLAMA_SERVER_BIN,
                    "--model",
                    model_path,
                    "--port",
                    str(port),
                    "--host",
                    "127.0.0.1",
                ]
                if is_embedding:
                    cmd.append("--embedding")

            # 5. Start process with logs redirected to a file
            model_safe = model_name.replace("/", "_").replace(":", "_")
            log_path = os.path.join(HERD_LOGS_DIR, f"{model_safe}.log")
            os.makedirs(HERD_LOGS_DIR, exist_ok=True)

            logger.info(
                f"Starting server for '{model_name}' on port {port}. Log: {log_path}"
            )
            log_file = open(log_path, "w")

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd, stdout=log_file, stderr=log_file
                )
            except Exception as e:
                log_file.close()
                logger.error(f"Failed to start process for '{model_name}': {e}")
                raise RuntimeError(
                    f"Failed to launch model server. Binary paths might be incorrect or missing: {e}"
                )

            # 6. Wait for the server to bind and respond on the port
            healthy = await self._wait_for_port(port)
            if not healthy:
                try:
                    process.terminate()
                    await process.wait()
                except Exception:
                    pass
                log_file.close()
                raise RuntimeError(
                    f"Model server '{model_name}' failed to bind on port {port} within timeout. "
                    f"Please check the logs at: {log_path}"
                )

            # Wait for the model weights to load fully
            ready = await self._wait_for_model_ready(port, is_whisper)
            if not ready:
                try:
                    process.terminate()
                    await process.wait()
                except Exception:
                    pass
                log_file.close()
                raise RuntimeError(
                    f"Model server '{model_name}' failed to load weights and become ready within timeout. "
                    f"Please check the logs at: {log_path}"
                )

            # 7. Register model server using resolved model path as key
            self.running_models[model_path] = {
                "process": process,
                "port": port,
                "last_accessed": time.time(),
                "log_file": log_file,
                "is_whisper": is_whisper,
                "is_embedding": is_embedding,
                "model_name": model_name,
                "model_path": model_path,
                "log_path": log_path,
                "idle_timeout": idle_timeout
                if idle_timeout is not None
                else IDLE_TIMEOUT,
            }

            return port

    async def stop_model(self, model_name: str):
        """Stops a running model server process."""
        try:
            model_path = resolve_model_path(model_name)
        except Exception:
            model_path = None
            for path, info in list(self.running_models.items()):
                if info["model_name"] == model_name:
                    model_path = path
                    break

        if model_path and model_path in self.running_models:
            info = self.running_models.pop(model_path)
            process = info["process"]
            log_file = info["log_file"]

            logger.info(
                f"Stopping model server '{model_name}' on port {info['port']}..."
            )
            try:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Process for '{model_name}' did not exit. Force killing..."
                    )
                    process.kill()
                    await process.wait()
            except ProcessLookupError:
                logger.info(
                    f"Model server '{model_name}' process already stopped."
                )
            except Exception as e:
                logger.error(f"Error terminating model server for '{model_name}': {e}")
            finally:
                try:
                    log_file.close()
                except Exception:
                    pass

    async def _wait_for_port(self, port: int, timeout: float = 30.0) -> bool:
        """Polls a TCP port until a connection can be established, or times out."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.close()
                await writer.wait_closed()
                return True
            except Exception:
                await asyncio.sleep(0.5)
        return False

    async def _wait_for_model_ready(self, port: int, is_whisper: bool, timeout: float = 120.0) -> bool:
        """Waits for the model server to finish loading and be fully ready to serve."""
        if is_whisper:
            # Whisper loads fast, port binding is sufficient
            return True

        start_time = time.time()
        url = f"http://127.0.0.1:{port}/health"
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < timeout:
                try:
                    response = await client.get(url, timeout=1.0)
                    if response.status_code == 200:
                        # Fully loaded and ready!
                        return True
                    elif response.status_code == 503:
                        # Still loading model
                        logger.info(f"Model on port {port} is still loading. Retrying...")
                    elif response.status_code == 404:
                        # If /health doesn't exist on this server version, fall back to assuming ready
                        logger.warning(f"Health endpoint not found (404) on port {port}. Assuming ready.")
                        return True
                except httpx.RequestError:
                    # Connection error or timeout
                    pass
                
                await asyncio.sleep(1.0)
                
        logger.warning(f"Timeout waiting for model on port {port} to become ready.")
        return False

    async def cleanup_loop(self):
        """Periodically checks and stops models that have been idle past the timeout."""
        while True:
            await asyncio.sleep(10)
            now = time.time()
            to_stop = []

            async with self.lock:
                for model_path, info in list(self.running_models.items()):
                    model_name = info["model_name"]
                    # Clean up dead processes first
                    if info["process"].returncode is not None:
                        logger.warning(
                            f"Model '{model_name}' server process terminated unexpectedly."
                        )
                        self.running_models.pop(model_path)
                        try:
                            info["log_file"].close()
                        except Exception:
                            pass
                        continue

                    # Calculate idle time
                    idle_time = now - info["last_accessed"]
                    model_idle_timeout = info.get("idle_timeout", IDLE_TIMEOUT)
                    if model_idle_timeout > 0 and idle_time > model_idle_timeout:
                        to_stop.append((model_name, idle_time))

            for model_name, idle_time in to_stop:
                logger.info(
                    f"Model '{model_name}' has been idle for {idle_time:.1f}s. Stopping process."
                )
                await self.stop_model(model_name)

    def get_process_resources(self, pid: int) -> dict:
        """Returns CPU percentage and RSS RAM usage in bytes for the process and its children."""
        if psutil is None:
            return {"cpu_percent": 0.0, "memory_bytes": 0}
        try:
            proc = psutil.Process(pid)
            mem = proc.memory_info().rss
            cpu = proc.cpu_percent(interval=None)

            # Sum up children resources recursively
            for child in proc.children(recursive=True):
                try:
                    mem += child.memory_info().rss
                    cpu += child.cpu_percent(interval=None)
                except Exception:
                    pass
            return {"cpu_percent": round(cpu, 1), "memory_bytes": mem}
        except Exception:
            return {"cpu_percent": 0.0, "memory_bytes": 0}
