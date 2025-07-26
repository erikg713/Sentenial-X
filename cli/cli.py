import asyncio
import json
import os
import typer
import requests

from config import (
    COMMAND_POLL_INTERVAL,
    AGENT_ID,
    NETWORK_PORT,
)
from memory import enqueue_command

app = typer.Typer(help="📡 Sentenial-X AI Agent CLI Interface")

@app.command()
def start():
    """
    🚀 Start the AI Agent in the foreground.
    """
    os.execvp("python3", ["python3", "agent.py"])

@app.command()
def send(cmd: str):
    """
    📤 Enqueue a command to the AI agent.
    """
    asyncio.run(enqueue_command(AGENT_ID, cmd))
    typer.echo(f"✅ Command queued: {cmd}")

@app.command()
def status():
    """
    🧠 Query agent's health endpoint.
    """
    url = f"http://localhost:{NETWORK_PORT}/health"
    try:
        resp = requests.get(url, timeout=3)
        resp.raise_for_status()
        typer.echo(json.dumps(resp.json(), indent=2))
    except Exception as e:
        typer.echo(f"❌ Error: {e}")

@app.command()
def broadcast(message: str):
    """
    📡 Manually broadcast a status message to peers.
    """
    url = f"http://localhost:{NETWORK_PORT}/message"
    payload = {"from": AGENT_ID, "msg": message}
    try:
        resp = requests.post(url, json=payload, timeout=3)
        if resp.status_code == 204:
            typer.echo("📢 Broadcast sent successfully.")
        else:
            typer.echo(f"⚠️ Unexpected response: {resp.status_code}")
    except Exception as e:
        typer.echo(f"❌ Broadcast error: {e}")

if __name__ == "__main__":
    app()