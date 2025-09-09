### Sentenial-X-A.I. CLI 🤖 ###
------------------------------------
A small, focused command-line interface to manage the Sentenial-X AI agent. Use the CLI to start/stop the agent, send commands, broadcast messages, and check status from a terminal or service manager.

Features
- Simple POSIX-friendly CLI with subcommands: start, stop, status, send, broadcast, and logs.
- Lightweight configuration via environment variables or a YAML file.
- Designed to run interactively or as a system service.

Table of contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Running as a service (systemd example)](#running-as-a-service-systemd-example)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License & contact](#license--contact)

Requirements
- Python 3.8+
- pip
- (Optional) Any API keys required by your agent (set as environment variables or in config)

Installation

From the repository (recommended for development)
1. Clone the repo:
   ```bash
   git clone https://github.com/erikg713/Sentenial-X.git
   cd Sentenial-X/sentenial_x_ai
   ```
2. Install:
   ```bash
   python -m pip install -e .
   ```
   The editable install is useful during development.

Install directly from GitHub (stable / non-editable)
```bash
pip install git+https://github.com/erikg713/Sentenial-X.git#subdirectory=sentenial_x_ai
```

Quick start
- Start the agent:
  ```bash
  sentenialx start
  ```
- Send a command to the agent:
  ```bash
  sentenialx send "reboot"
  ```
- Broadcast a message to connected clients:
  ```bash
  sentenialx broadcast "Agent online"
  ```
- Check status:
  ```bash
  sentenialx status
  ```
- Stop the agent:
  ```bash
  sentenialx stop
  ```

Usage and CLI reference
Run help to see all available commands and options:
```bash
sentenialx --help
# or for subcommands
sentenialx start --help
sentenialx send --help
```

Common subcommands
- start — launch the agent (foreground by default; use systemd for background runs)
- stop — gracefully stop the running agent
- status — show agent status and PID if running
- send <message> — send a single command/message to the agent
- broadcast <message> — broadcast a message to all connected clients
- logs — show recent logs (if implemented)

Configuration
Sentenial-X supports both environment variables and a YAML configuration file. The CLI will check environment variables first, then the config file if present.

Suggested environment variables
- SENTENIALX_CONFIG — path to YAML config file (default: ./sentenialx.yml)
- SENTENIALX_LOG_LEVEL — DEBUG, INFO, WARNING, ERROR (default: INFO)
- SENTENIALX_API_KEY — for any external services used by the agent

Example minimal YAML config (sentenialx.yml)
```yaml
agent:
  name: SentenialX
  port: 8080

logging:
  level: INFO
  file: /var/log/sentenialx/sentenialx.log
```

Running as a service (systemd example)
Create /etc/systemd/system/sentenialx.service:
```ini
[Unit]
Description=Sentenial-X Agent
After=network.target

[Service]
Type=simple
User=sentenialx
WorkingDirectory=/opt/sentenialx
ExecStart=/usr/bin/sentenialx start
Restart=on-failure
Environment=PYTHONUNBUFFERED=1
# Optionally set SENTENIALX_CONFIG or other env vars here:
# Environment="SENTENIALX_CONFIG=/etc/sentenialx/config.yml"

[Install]
WantedBy=multi-user.target
```
Then reload systemd and enable:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now sentenialx.service
```

Development
- Run code style checks and tests before pushing changes.
- Use an editable install for development:
  ```bash
  python -m pip install -e .
  ```
- To run tests (if present):
  ```bash
  pytest
  ```

Troubleshooting
- Agent will not start:
  - Check logs (sentenialx logs or systemd journal).
  - Ensure config file and environment variables are set correctly.
  - Verify required ports are free.
- Commands not delivered:
  - Confirm the agent is running and listening on configured port.
  - Check network/firewall settings.
- Permission errors:
  - Running as a system service? Ensure the service user has proper file permissions for logs and data directories.

Contributing
Contributions are welcome. Please:
1. Open an issue describing the problem or feature.
2. Create a branch for your work: feature/<short-desc>.
3. Add tests for new functionality.
4. Open a pull request with a clear description of your changes.

License & contact
- License: Add your license file (LICENSE) in the repository root and update this section.
- Maintainer: erikg713 — https://github.com/erikg713

Acknowledgements
- Inspired by small, single-binary CLI tools for managing long-running agents.

Changelog
- See CHANGELOG.md (create one if you don't have it yet) for release notes.

```
