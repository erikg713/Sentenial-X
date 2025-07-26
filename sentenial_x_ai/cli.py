import asyncio
import json
import os
import typer
import requests
import logging

from memory import enqueue_command
from config import COMMAND_POLL_INTERVAL, AGENT_ID, NETWORK_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create Typer app
app = typer.Typer(help="📡 Sentenial-X AI Agent CLI Interface")


@app.command()
def start():
    """
    🚀 Start the AI Agent in the foreground.
    """
    typer.echo("🚀 Launching Sentenial-X agent...")
    os.execvp("python3", ["python3", "agent.py"])


@app.command()
def send(command: str):
    """
    📤 Enqueue a command to the AI agent.
    """
    try:
        asyncio.run(enqueue_command(AGENT_ID, command.strip()))
        typer.echo(f"✅ Command queued: {command}")
    except Exception as e:
        logging.exception("❌ Failed to enqueue command")
        typer.echo(f"❌ Error queuing command: {e}")


@app.command()
def status():
    """
    🧠 Check the health status of the AI agent.
    """
    url = f"http://localhost:{NETWORK_PORT}/health"
    try:
        response = requests.get(url, timeout=3)
        response.raise_for_status()
        typer.echo(json.dumps(response.json(), indent=2))
    except requests.ConnectionError:
        typer.echo("❌ Agent is not running or unreachable.")
    except requests.Timeout:
        typer.echo("⏱️ Request to agent timed out.")
    except Exception as e:
        logging.exception("❌ Unexpected error during status check")
        typer.echo(f"❌ Unexpected error: {e}")


@app.command()
def broadcast(message: str):
    """
    📡 Manually broadcast a status message to peer agents.
    """
    url = f"http://localhost:{NETWORK_PORT}/message"
    payload = {"from": AGENT_ID, "msg": message}

    try:
        response = requests.post(url, json=payload, timeout=3)
        if response.status_code == 204:
            typer.echo("📢 Broadcast sent successfully.")
        else:
            typer.echo(f"⚠️ Unexpected response: {response.status_code}")
    except requests.RequestException as e:
        logging.exception("❌ Broadcast request failed")
        typer.echo(f"❌ Broadcast error: {e}")


@app.command()
def stop():
    """
    🛑 Stop the AI Agent gracefully.
    (Note: requires agent to support signal handling or PID tracking.)
    """
    try:
        os.system("pkill -f agent.py")
        typer.echo("🛑 Agent stopped.")
    except Exception as e:
        logging.exception("❌ Failed to stop agent")
        typer.echo(f"❌ Error stopping agent: {e}")


# Entry point
if __name__ == "__main__":
    app()
