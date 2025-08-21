# Sentenial‑X CLI – Full Operator Guide

Overview

Sentenial‑X is an advanced AI-driven cybersecurity platform. Its CLI suite allows operators to:

Detect adversarial AI attacks (e.g., WormGPT-style)

Scan for blind spots in detection

Perform NLP-based threat analysis

Stream real-time telemetry

Dispatch alerts

Run orchestrator commands

Simulate threat scenarios


This guide covers installation, usage, and all available commands.


---

Installation

# Clone repository
git clone https://github.com/erikg713/Sentenial-X.git
cd Sentenial-X

# Optional: create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Ensure all modules (sentenial_core, memory, config, telemetry, orchestrator, etc.) are available


---

CLI Usage

./sentenial_cli_full.py <command> [options]

Run --help for general guidance:

./sentenial_cli_full.py --help


---

Commands & Examples

1. WormGPT Detector

Description: Analyze adversarial AI inputs and generate countermeasures.

./sentenial_cli_full.py wormgpt-detector -p "malicious prompt example" -t 0.8

Options:

-p / --prompt: Adversarial input text (required)

-t / --temperature: Exploration randomness (default: 0.7)



---

2. Blind Spot Tracker

Description: Scan for detection blind spots in your system.

./sentenial_cli_full.py blindspots

No additional arguments required.


---

3. Cortex NLP Threat Analysis

Description: Run NLP-based analysis on log data or network events.

./sentenial_cli_full.py cortex -s "/var/log/syslog" -f "error OR failed"

Options:

-s / --source: Log file or source (required)

-f / --filter: Optional filter expression for targeted analysis



---

4. Orchestrator Commands

Description: Execute central commands via the orchestrator engine.

./sentenial_cli_full.py orchestrator -a "update_policy" -p '{"policy_id": "123"}'

Options:

-a / --action: Orchestrator action name (required)

-p / --params: JSON dictionary of parameters (optional)



---

5. Real-Time Telemetry Streaming

Description: Stream live telemetry from sensors, agents, or logs.

./sentenial_cli_full.py telemetry -s "network_monitor" -f "high_severity"

Options:

-s / --source: Telemetry source (required)

-f / --filter: Optional filter expression



---

6. Alert Dispatcher

Description: Dispatch alerts to operators or systems.

./sentenial_cli_full.py alert -t "ransomware_detected" -s "high"

Options:

-t / --type: Alert type (required)

-s / --severity: Severity level (default: medium)



---

7. Threat Simulator

Description: Run red-team style threat simulations.

./sentenial_cli_full.py simulate -sc "phishing_campaign"

Options:

-sc / --scenario: Name of the threat scenario (required)



---

Logging & Memory

All CLI actions automatically log results into the Sentenial-X memory system:

Stored in SQLite or configured memory backend

Includes timestamp, action type, parameters, and results

Supports audit and analysis for compliance



---

Extending the CLI

You can extend this CLI to add new modules:

1. Create a new async handler function.


2. Add a new subparser in main().


3. Call your module and log results via enqueue_command.




---

Recommended Best Practices

Run CLI inside a virtual environment.

Keep config.py secure, especially AGENT_ID and credentials.

Monitor telemetry in real-time for critical alerts.

Use simulate before deploying new detection policies.

Automate routine scans using cron or task scheduler.



---
