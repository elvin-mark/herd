import os
import shutil

# Root directory
HERD_HOME = os.environ.get("HERD_HOME", os.path.expanduser("~/.herd"))
HERD_MODELS_DIR = os.path.join(HERD_HOME, "models")
HERD_LOGS_DIR = os.path.join(HERD_HOME, "logs")

# Port for Herd API gateway
HERD_PORT = int(os.environ.get("HERD_PORT", "11434"))

# Idle timeout for model servers in seconds
IDLE_TIMEOUT = int(os.environ.get("HERD_IDLE_TIMEOUT", "300"))

# Binary paths
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN") or shutil.which("llama-server")
WHISPER_SERVER_BIN = os.environ.get("WHISPER_SERVER_BIN") or shutil.which(
    "whisper-server"
)

# Ensure directories exist
os.makedirs(HERD_HOME, exist_ok=True)
os.makedirs(HERD_MODELS_DIR, exist_ok=True)
os.makedirs(HERD_LOGS_DIR, exist_ok=True)
