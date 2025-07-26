# tests/test_cli.py
from typer.testing import CliRunner
from sentenialx.cli import app

runner = CliRunner()

def test_status():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
