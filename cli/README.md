# Sentenial-X AI Agent (cli/)

An adaptive, asyncio-powered AI agent that learns from experience, responds to commands, broadcasts status, and supports plugins. This README covers the CLI wrapper and how to interact with the local agent.

Key highlights:
- Async command ingestion and persisted event memory (SQLite)
- Heartbeat & broadcasting across peers
- Plugin system for adding custom commands
- Simple ML classifier for categorization
- Small, focused CLI to start, query, and send commands to the agent

---

## Repository

Clone the repository (use your fork/branch when contributing):

```bash
git clone https://github.com/erikg713/Sentenial-X.git
cd Sentenial-X/cli
```

---

## Quickstart (recommended)

1. Create and activate a venv (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (project root):

```bash
pip install -r requirements.txt
```

3. Copy or create a `.env` file in the project root (example below).

---

## Example .env

Place this at the repository root (adjust as needed):

```ini
DB_PATH=./data/agent.db
AGENT_ID=sentenial-x-ai-bot
NETWORK_HOST=0.0.0.0
NETWORK_PORT=8000
PEERS=http://localhost:8001,http://localhost:8002
HEARTBEAT_INTERVAL=10
COMMAND_POLL_INTERVAL=5
STRATEGY_INTERVAL=15
BROADCAST_INTERVAL=30
LOG_LEVEL=INFO
```

Notes:
- DB_PATH is relative to the repository root by default; make sure the `data/` directory exists or update the path.
- NETWORK_HOST and NETWORK_PORT are used by the HTTP endpoints.

---

## CLI: Usage

From the `cli/` directory:

```bash
source .venv/bin/activate
python cli.py --help
```

Commands:
- start    — Launch the agent in the foreground (execs the interpreter so logs appear in your terminal)
- send     — Enqueue a command for the agent (async, persisted)
- status   — Query the agent's `/health` endpoint and pretty-print the JSON result
- broadcast— Send a manual status message to the agent's `/message` endpoint

Each command supports a `--verbose` flag for extra logging and a `--host/--port` override for local testing.

Examples:

- Start the agent (uses the same Python interpreter as the CLI):
```bash
python cli.py start
```

- Enqueue a command:
```bash
python cli.py send "analyze latest logs"
```

- Check health:
```bash
python cli.py status --host localhost --port 8000
```

- Broadcast a status message:
```bash
python cli.py broadcast "All systems nominal" --host localhost --port 8000
```

---

## Plugins

Drop modules into `plugins/`. Each plugin should expose a `register` function:

```python
def register(register_fn):
    register_fn('command_name', handler_fn)

def handler_fn(arg: str) -> str:
    # perform action and return a string result
    return "result"
```

The agent loads `plugins/` on startup and registers the provided commands.

---

## Running as a Service (systemd)

Example unit (adjust paths and username):

/etc/systemd/system/sentenial-x.service
```ini
[Unit]
Description=Sentenial-X AI Agent
After=network.target

[Service]
Type=simple
WorkingDirectory=/path/to/your/repo
ExecStart=/path/to/venv/bin/python agent.py
Restart=on-failure
User=youruser
EnvironmentFile=/path/to/your/repo/.env

[Install]
WantedBy=multi-user.target
```

Enable & start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable sentenial-x
sudo systemctl start sentenial-x
```

---

## Troubleshooting

- If `status` or `broadcast` fail, make sure the agent is running and the host/port are correct.
- If database errors occur, ensure `DB_PATH` points to a writable location and the parent directory exists.
- For networking between peers, ensure firewalls allow the configured ports.

---

## Testing & CI

- Unit tests live under `tests/`.
- CI runs mypy, linters, and pytest — keep code typed and well-documented.

---

## Contributing

1. Fork, create a feature branch.
2. Add tests under `tests/` and keep changes focused.
3. Open a PR and reference the issue(s).

Please follow the repository's code style and typing guidelines.

---

## License

MIT
```   ```bash
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
