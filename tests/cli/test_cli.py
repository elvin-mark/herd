from typer.testing import CliRunner

from herd.cli import app

runner = CliRunner()


def test_cli_help():
    """Verifies that running 'herd --help' executes cleanly and displays command panels."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Core Interfaces" in result.stdout
    assert "Model Management" in result.stdout


def test_cli_doctor():
    """Verifies that 'herd doctor' runs hardware diagnostics."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "System Doctor" in result.stdout
