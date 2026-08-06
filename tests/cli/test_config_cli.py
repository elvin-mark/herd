import os
import tempfile

from typer.testing import CliRunner

from herd.commands.config import config_app

runner = CliRunner()


def test_config_show_cli(monkeypatch):
    """Verifies that 'herd config show' runs cleanly and outputs configuration table."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_config = os.path.join(tmp_dir, "config.json")
        monkeypatch.setattr("herd.commands.config.CONFIG_FILE", fake_config)

        result = runner.invoke(config_app, ["show"])
        assert result.exit_code == 0
        assert "Herd Configurations" in result.stdout


def test_config_set_and_provider_cli(monkeypatch):
    """Verifies setting key-value pairs and cloud provider settings in config.json."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_config = os.path.join(tmp_dir, "config.json")
        monkeypatch.setattr("herd.commands.config.CONFIG_FILE", fake_config)
        monkeypatch.setattr("herd.core.config.CONFIG_FILE", fake_config)

        # 1. Set key
        res_set = runner.invoke(config_app, ["set", "default_llm", "test-model-id"])
        assert res_set.exit_code == 0
        assert "Success!" in res_set.stdout

        # 2. Set provider
        res_prov = runner.invoke(
            config_app, ["set-provider", "openai", "--api-key", "sk-test123456789"]
        )
        assert res_prov.exit_code == 0
        assert "Configured provider" in res_prov.stdout

        # 3. Remove provider
        res_rem = runner.invoke(config_app, ["remove-provider", "openai"])
        assert res_rem.exit_code == 0
        assert "Removed provider" in res_rem.stdout
