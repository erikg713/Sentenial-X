## 💻 Sentenial‑X CLI — Full Operator Guide ##

Overview

Sentenial‑X is an AI-driven cybersecurity platform. Its CLI provides operators the ability to:

Detect adversarial AI attacks (e.g., WormGPT-style)

Track blind spots in detection

Perform NLP-based threat analysis

Stream live telemetry

Dispatch alerts

Execute orchestrator actions

Simulate cyber-attack scenarios

Manage AI and endpoint agents



---

Installation

# Clone repository
git clone https://github.com/erikg713/Sentenial-X.git
cd Sentenial-X

# Optional: create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

> Ensure core modules (memory.py, config.py, telemetry.py, orchestrator.py, agent*.py, etc.) are available.




---

CLI Usage

./sentenial_cli_full.py <command> [options]
./sentenial_cli_full.py --help  # show help

All commands support:

--json       # output machine-readable JSON
--verbose    # detailed logging
-h, --help   # command-specific help


---

Commands

1. WormGPT Detector

Analyze adversarial AI inputs.

./sentenial_cli_full.py wormgpt-detector -p "malicious prompt" -t 0.8

-p, --prompt → input text (required)

-t, --temperature → randomness, default 0.7



---

2. Blind Spot Tracker

Scan for detection blind spots.

./sentenial_cli_full.py blindspots

No additional arguments required.


---

3. Cortex NLP Threat Analysis

Run NLP-based log or network analysis.

./sentenial_cli_full.py cortex -s "/var/log/syslog" -f "error OR failed"

-s, --source → log file/source (required)

-f, --filter → optional filter expression



---

4. Orchestrator

Execute central commands.

./sentenial_cli_full.py orchestrator -a "update_policy" -p '{"policy_id":"123"}'

-a, --action → action name (required)

-p, --params → JSON dictionary (optional)



---

5. Telemetry Streaming

Stream live telemetry from sensors or agents.

./sentenial_cli_full.py telemetry -s "network_monitor" -f "high_severity"

-s, --source → telemetry source (required)

-f, --filter → optional filter expression



---

6. Alert Dispatcher

Dispatch alerts to operators or systems.

./sentenial_cli_full.py alert -t "ransomware_detected" -s "high"

-t, --type → alert type (required)

-s, --severity → default medium



---

7. Threat Simulator

Run red-team style simulations.

./sentenial_cli_full.py simulate -sc "phishing_campaign"

-sc, --scenario → scenario name (required)



---

8. Agent Management

Start, stop, restart, and check the status of AI and endpoint agents.

# Start an AI agent
python -m cli.agent --start ai_agent

# Stop the retaliation bot
python -m cli.agent --stop retaliation_bot

# Restart the endpoint agent
python -m cli.agent --restart endpoint_agent

# Show status of all agents
python -m cli.agent --status

# Show status of a specific agent
python -m cli.agent --status ai_agent


---

Logging & Memory

All CLI actions log results in SQLite (or configured memory backend)

Stored info: timestamp, action type, parameters, results

Supports auditing and compliance


Example ad-hoc query:

-- Last 24h high-severity alerts
SELECT * FROM alerts
WHERE severity='high' AND timestamp >= datetime('now','-1 day')
ORDER BY timestamp DESC;


---

Best Practices

Run CLI inside a virtual environment

Keep config.py secure (AGENT_ID, tokens, credentials)

Monitor telemetry in real-time

Simulate before deploying new detection policies

Automate routine scans via cron or task scheduler



---

Extending the CLI

1. Create an async handler function in your module.


2. Add a subparser in sentenial_cli_full.py.


3. Call your module and log results via enqueue_command or memory helper.


4. Return JSON-serializable results (action, params, result, timestamp).



Example stub:

async def handle_mycmd(args):
    from my_module import run
    params = {"foo": args.foo}
    result = await run(params)
    from memory import write_command
    write_command(action="mycmd", params=params, result=result)
    print(result)


---
