import os
import tempfile

from herd.services.agent import (
    agent_list_dir,
    agent_read_file,
    agent_search_grep,
    agent_view_file_lines,
    agent_write_file,
)


def test_agent_file_operations():
    """Verifies agent_write_file, agent_read_file, agent_view_file_lines, and agent_list_dir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, "test.txt")
        content = "Line 1: Hello\nLine 2: World\nLine 3: Herd\n"

        # 1. Write
        res_write = agent_write_file(file_path, content)
        assert "successfully written" in res_write.lower()

        # 2. Read
        res_read = agent_read_file(file_path)
        assert res_read == content

        # 3. View Lines
        res_lines = agent_view_file_lines(file_path, start_line=2, end_line=3)
        assert "2: Line 2: World" in res_lines
        assert "3: Line 3: Herd" in res_lines

        # 4. List Dir
        res_dir = agent_list_dir(tmp_dir)
        assert "test.txt" in res_dir

        # 5. Grep Search
        res_grep = agent_search_grep("Herd", path=tmp_dir)
        assert "Line 3: Herd" in res_grep
