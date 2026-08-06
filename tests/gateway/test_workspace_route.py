import os
import tempfile

from fastapi.testclient import TestClient

from herd.api.server import app

client = TestClient(app)


def test_list_workspace_files():
    """Verifies listing workspace files route."""
    response = client.get("/v1/workspace/files")
    assert response.status_code == 200
    data = response.json()
    assert "workspace" in data
    assert "files" in data
    assert isinstance(data["files"], list)


def test_workspace_file_security_jailing(monkeypatch):
    """Verifies path jailing blocks attempts to access files outside workspace boundary."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        real_tmp = os.path.realpath(tmp_dir)
        monkeypatch.setattr("herd.api.routers.workspace.get_workspace_dir", lambda: real_tmp)

        # 1. Create file inside workspace
        inside_file = os.path.join(tmp_dir, "test.txt")
        with open(inside_file, "w") as f:
            f.write("inside workspace")

        res_ok = client.get("/v1/workspace/file?path=test.txt")
        assert res_ok.status_code == 200
        assert res_ok.text == "inside workspace"

        # 2. Attempt path traversal outside workspace
        res_jail = client.get("/v1/workspace/file?path=../../etc/passwd")
        assert res_jail.status_code == 403
        assert "Access denied" in res_jail.json()["error"]
