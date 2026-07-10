import os
import sys

# Ensure local herd package is loaded from the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from herd.services.agent import agent_view_file_lines

# Create a temporary dummy file for testing
dummy_path = "tmp_test_agent_file.txt"
with open(dummy_path, "w") as f:
    f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

try:
    # 1. Test view_file_lines extraction and indexing
    res = agent_view_file_lines(dummy_path, 2, 4)
    expected = "2: Line 2\n3: Line 3\n4: Line 4\n"
    assert res == expected, f"Expected '{expected}', got '{res}'"
    print("Success: view_file_lines extraction and indexing is correct.")

    # 2. Test view_file_lines out of bounds validation
    res_err = agent_view_file_lines(dummy_path, 10, 12)
    assert "Error:" in res_err, f"Expected error, got '{res_err}'"
    print("Success: view_file_lines correctly reports out-of-bounds error.")

finally:
    if os.path.exists(dummy_path):
        os.remove(dummy_path)

print("All agent tool tests completed successfully!")
