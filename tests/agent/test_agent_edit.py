import os
import tempfile

from herd.services.agent import agent_edit_file


def test_agent_edit_file_exact():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("def foo():\n    return 42\n")
        path = f.name

    try:
        res = agent_edit_file(path, "return 42", "return 100")
        assert "exact match" in res
        with open(path) as f:
            content = f.read()
        assert "return 100" in content
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_agent_edit_file_fuzzy_whitespace():
    with tempfile.NamedTemporaryFile("w+", delete=False) as f:
        f.write("def bar():\n    x = 10\n    return x\n")
        path = f.name

    try:
        target = "def bar():\n\tx = 10\n\treturn x"
        replacement = "def bar():\n    return 999"
        res = agent_edit_file(path, target, replacement)
        assert "Successfully edited file" in res
        with open(path) as f:
            content = f.read()
        assert "return 999" in content
    finally:
        if os.path.exists(path):
            os.remove(path)
