import json
import os
import tempfile

from herd.services.manager import ProcessManager


def test_process_manager_state_serialization():
    """Verifies that active process state is written and read from state JSON file."""
    pm = ProcessManager()

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = os.path.join(tmp_dir, "active_processes.json")
        pm.running_models = {
            "/tmp/fake_model.gguf": {
                "model_name": "fake_model",
                "port": 42000,
                "is_whisper": False,
                "is_embedding": False,
                "process": type("Proc", (), {"pid": 99999})(),
            }
        }

        pm._save_active_processes_sync(state_file)

        assert os.path.exists(state_file)
        with open(state_file, "r") as f:
            data = json.load(f)

        assert "/tmp/fake_model.gguf" in data
        assert data["/tmp/fake_model.gguf"]["port"] == 42000
        assert data["/tmp/fake_model.gguf"]["pid"] == 99999


def test_process_manager_orphan_cleanup_nonexistent():
    """Verifies that orphan cleanup handles stale non-existent PIDs safely without throwing errors."""
    pm = ProcessManager()

    with tempfile.TemporaryDirectory() as tmp_dir:
        state_file = os.path.join(tmp_dir, "active_processes.json")
        stale_data = {
            "/tmp/stale.gguf": {
                "model_name": "stale",
                "port": 42001,
                "pid": 9999999,  # Non-existent PID
            }
        }
        with open(state_file, "w") as f:
            json.dump(stale_data, f)

        pm._cleanup_orphan_processes(state_file)
        assert os.path.exists(state_file)
