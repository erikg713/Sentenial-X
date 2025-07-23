# Sentenial-X AI Agent

An adaptive, asyncio-powered AI agent that learns from its experience,  
responds to commands, broadcasts status, and supports plugins.  

---

## Features

- Heartbeat logging  
- Command ingestion & execution  
- Memory of events (SQLite backend)  
- Simple ML classifier (Naive Bayes)  
- Plugin system for custom commands  
- Async HTTP server for inter-agent messaging  
- Command-line interface for interactive use  

---

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/yourorg/sentenial_x_ai.git
   cd sentenial_x_ai
   ```
2. Create a Python virtualenv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Copy or create an `.env` file in project root:

```ini
DB_PATH=./data/agent.db
AGENT_ID=sentenial-x-ai-bot
NETWORK_PORT=8000
PEERS=http://localhost:8001,http://localhost:8002
HEARTBEAT_INTERVAL=10
COMMAND_POLL_INTERVAL=5
STRATEGY_INTERVAL=15
BROADCAST_INTERVAL=30
```

---

## Usage

### As a Service
```bash
sudo cp sentenial-threat-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable sentenial-threat-monitor
sudo systemctl start sentenial-threat-monitor
```

### Command-Line Interface
Interact with your agent locally:

```bash
source .venv/bin/activate
python cli.py --help
```

---

## CLI Commands

| Command    | Description                             |
|------------|-----------------------------------------|
| `start`    | Launch the agent in the foreground      |
| `send`     | Enqueue a command to the agent          |
| `status`   | Query `/health` endpoint                |
| `broadcast`| Manually broadcast a status message     |

---

## Plugins

Drop your modules into `plugins/`. Each must define:

```python
def register(register_fn):
    register_fn('command_name', handler_fn)

def handler_fn(arg: str) -> str:
    return "result"
```

---

## Contributing

1. Fork & branch.  
2. Add tests under `tests/`.  
3. Submit a PR; CI runs `mypy`, lint, pytest.  

---

## License

MIT  
```

---

### 4.2 `cli.py`

```python
#!/usr/bin/env python3
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
```

---
