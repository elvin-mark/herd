import os
import shutil
import json


class HerdSettings:
    def __init__(self):
        self.home = os.environ.get("HERD_HOME", os.path.expanduser("~/.herd"))
        self.models_dir = os.path.join(self.home, "models")
        self.logs_dir = os.path.join(self.home, "logs")
        self.config_file = os.path.join(self.home, "config.json")

        # Load global overrides
        self.global_overrides = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    self.global_overrides = json.load(f)
            except Exception:
                pass

        # Load local overrides (.herd.json) by searching upwards
        self.local_overrides = {}
        try:
            current_dir = os.getcwd()
            while True:
                local_config_path = os.path.join(current_dir, ".herd.json")
                if os.path.exists(local_config_path):
                    try:
                        with open(local_config_path, "r") as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                self.local_overrides = data
                    except Exception:
                        pass
                    break

                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:
                    break
                current_dir = parent_dir
        except Exception:
            pass

        # Merged overrides (local takes precedence)
        self.overrides = {}
        self.overrides.update(self.global_overrides)
        self.overrides.update(self.local_overrides)

        # Host and Port
        self.host = os.environ.get("HERD_HOST", "127.0.0.1")
        self.port = int(os.environ.get("HERD_PORT", "11434"))

        # Idle timeout
        self.idle_timeout = int(os.environ.get("HERD_IDLE_TIMEOUT", "300"))

        # Binary paths
        self.llama_server_bin = (
            os.environ.get("LLAMA_SERVER_BIN")
            or self.overrides.get("LLAMA_SERVER_BIN")
            or shutil.which("llama-server")
        )
        self.whisper_server_bin = (
            os.environ.get("WHISPER_SERVER_BIN")
            or self.overrides.get("WHISPER_SERVER_BIN")
            or shutil.which("whisper-server")
        )

        # Default model settings
        self.default_llm = self.overrides.get("default_llm")
        self.default_embedding = self.overrides.get("default_embedding")
        self.default_whisper = self.overrides.get("default_whisper")
        self.providers = self.overrides.get("providers", {})

        # Compiled commits
        self.llama_commit = self.overrides.get("LLAMA_COMMIT")
        self.whisper_commit = self.overrides.get("WHISPER_COMMIT")

        # Make sure directories exist
        os.makedirs(self.home, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def reload(self):
        self.__init__()

    def save(self):
        with open(self.config_file, "w") as f:
            json.dump(self.global_overrides, f, indent=4)


settings = HerdSettings()

# Export compatible module-level constants
HERD_HOME = settings.home
HERD_MODELS_DIR = settings.models_dir
HERD_LOGS_DIR = settings.logs_dir
CONFIG_FILE = settings.config_file

HERD_HOST = settings.host
HERD_PORT = settings.port
IDLE_TIMEOUT = settings.idle_timeout

LLAMA_SERVER_BIN = settings.llama_server_bin
WHISPER_SERVER_BIN = settings.whisper_server_bin

DEFAULT_LLM = settings.default_llm
DEFAULT_EMBEDDING = settings.default_embedding
DEFAULT_WHISPER = settings.default_whisper
PROVIDERS = settings.providers

LLAMA_COMMIT = settings.llama_commit
WHISPER_COMMIT = settings.whisper_commit


def load_config() -> dict:
    settings.reload()
    return settings.overrides


def save_config(config: dict):
    # Only update global overrides for things explicitly passed in the config dict
    for k, v in config.items():
        settings.global_overrides[k] = v

    settings.default_llm = config.get("default_llm", settings.default_llm)
    settings.default_embedding = config.get(
        "default_embedding", settings.default_embedding
    )
    settings.default_whisper = config.get("default_whisper", settings.default_whisper)
    if "LLAMA_SERVER_BIN" in config:
        settings.llama_server_bin = config["LLAMA_SERVER_BIN"]
    if "WHISPER_SERVER_BIN" in config:
        settings.whisper_server_bin = config["WHISPER_SERVER_BIN"]
    if "LLAMA_COMMIT" in config:
        settings.llama_commit = config["LLAMA_COMMIT"]
    if "WHISPER_COMMIT" in config:
        settings.whisper_commit = config["WHISPER_COMMIT"]
    if "providers" in config:
        settings.providers = config["providers"]
    settings.save()
