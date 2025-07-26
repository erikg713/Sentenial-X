import asyncio
import json
import os
import typer
import requests
from config import COMMAND_POLL_INTERVAL, AGENT_ID, NETWORK_PORT
from memory import enqueue_command

app = typer.Typer(help="Sentenial-X AI Agent CLI")

@app.command()
def start():
    """
    Start the agent in the foreground.
    """
    os.execvp("python3", ["python3", "agent.py"])

@app.command()
def send(cmd: str):
    """
    Enqueue a command for the agent.
    """
    asyncio.run(enqueue_command(AGENT_ID, cmd))
    typer.echo(f"Command queued: {cmd}")

@app.command()
def status():
    """
    Query the agent's health endpoint.
    """
    url = f"http://localhost:{NETWORK_PORT}/health"
    resp = requests.get(url)
    if resp.ok:
        data = resp.json()
        typer.echo(json.dumps(data, indent=2))
    else:
        typer.echo(f"Failed to reach agent: {resp.status_code}")

@app.command()
def broadcast(message: str):
    """
    Manually broadcast a status message.
    """
    url = f"http://localhost:{NETWORK_PORT}/message"
    payload = {"from": AGENT_ID, "msg": message}
    resp = requests.post(url, json=payload)
    if resp.status_code == 204:
        typer.echo("Broadcast sent.")
    else:
        typer.echo(f"Error: {resp.status_code}")

if __name__ == "__main__":
    app()