import os
import shutil
import json

# Root directory
HERD_HOME = os.environ.get("HERD_HOME", os.path.expanduser("~/.herd"))
HERD_MODELS_DIR = os.path.join(HERD_HOME, "models")
HERD_LOGS_DIR = os.path.join(HERD_HOME, "logs")

# Load local config.json overrides if present
CONFIG_FILE = os.path.join(HERD_HOME, "config.json")
config_overrides = {}
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            config_overrides = json.load(f)
    except Exception:
        pass

# Host and Port for Herd API gateway
HERD_HOST = os.environ.get("HERD_HOST", "127.0.0.1")
HERD_PORT = int(os.environ.get("HERD_PORT", "11434"))

# Idle timeout for model servers in seconds
IDLE_TIMEOUT = int(os.environ.get("HERD_IDLE_TIMEOUT", "300"))

# Binary paths
LLAMA_SERVER_BIN = (
    os.environ.get("LLAMA_SERVER_BIN")
    or config_overrides.get("LLAMA_SERVER_BIN")
    or shutil.which("llama-server")
)
WHISPER_SERVER_BIN = (
    os.environ.get("WHISPER_SERVER_BIN")
    or config_overrides.get("WHISPER_SERVER_BIN")
    or shutil.which("whisper-server")
)

# Ensure directories exist
os.makedirs(HERD_HOME, exist_ok=True)
os.makedirs(HERD_MODELS_DIR, exist_ok=True)
os.makedirs(HERD_LOGS_DIR, exist_ok=True)


def load_config() -> dict:
    """Loads current config settings from config.json."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_config(config: dict):
    """Saves config settings back to config.json."""
    os.makedirs(HERD_HOME, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


# Default model configurations
DEFAULT_LLM = config_overrides.get("default_llm")
DEFAULT_EMBEDDING = config_overrides.get("default_embedding")
DEFAULT_WHISPER = config_overrides.get("default_whisper")

