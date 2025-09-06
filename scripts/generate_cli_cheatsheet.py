# scripts/generate_cli_cheatsheet.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
# -*- coding: utf-8 -*-
"""
Generate a CLI cheatsheet for Sentenial-X
-----------------------------------------

- Lists all available simulators
- Lists playbooks
- Lists EmulationManager actions
- Outputs formatted CLI-friendly cheatsheet
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from core.simulator import (
    discover_simulators,
    EmulationManager,
    TelemetryCollector,
)
from core.simulator.attack_playbook import create_playbook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentenialX.CLI.Cheatsheet")


def generate_cheatsheet() -> Dict[str, Any]:
    """Generate structured cheatsheet information."""
    cheatsheet: Dict[str, Any] = {}

    # Step 1: Discover simulators
    simulators = discover_simulators()
    cheatsheet["simulators"] = [sim.name for sim in simulators]

    # Step 2: Sample playbook
    playbook = create_playbook()
    cheatsheet["playbooks"] = [{"id": playbook.id, "name": playbook.name, "steps": [s.id for s in playbook.steps]}]

    # Step 3: EmulationManager actions
    cheatsheet["emulation_manager"] = ["register(simulator)", "start_all()", "stop_all()", "run_all(sequential=True)", "collect_telemetry()"]

    # Step 4: TelemetryCollector actions
    cheatsheet["telemetry_collector"] = ["add(telemetry)", "report(last_n=None)", "summary()"]

    return cheatsheet


def print_cheatsheet(cheatsheet: Dict[str, Any]) -> None:
    """Print a human-readable CLI cheatsheet."""
    print("\n==== Sentenial-X CLI Cheatsheet ====\n")

    print("Simulators:")
    for sim in cheatsheet.get("simulators", []):
        print(f"  - {sim}")

    print("\nAttack Playbooks:")
    for pb in cheatsheet.get("playbooks", []):
        steps = ", ".join(pb.get("steps", []))
        print(f"  - {pb['name']} (id: {pb['id']}) -> Steps: {steps}")

    print("\nEmulationManager Actions:")
    for action in cheatsheet.get("emulation_manager", []):
        print(f"  - {action}")

    print("\nTelemetryCollector Actions:")
    for action in cheatsheet.get("telemetry_collector", []):
        print(f"  - {action}")

    print("\n===================================\n")


def main(output_json: bool = False) -> None:
    cheatsheet = generate_cheatsheet()
    if output_json:
        print(json.dumps(cheatsheet, indent=2))
    else:
        print_cheatsheet(cheatsheet)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a CLI cheatsheet for Sentenial-X.")
    parser.add_argument("--json", action="store_true", help="Output cheatsheet as JSON")
    args = parser.parse_args()

    main(output_json=args.json)
CLI_COMMANDS = [
    ("wormgpt-detector", "Analyze adversarial AI inputs", 'sentenial_cli_full.py wormgpt-detector -p "malicious prompt"'),
    ("blindspots", "Scan for detection blind spots", "sentenial_cli_full.py blindspots"),
    ("cortex", "Run NLP-based threat analysis", 'sentenial_cli_full.py cortex -s "/var/log/syslog" -f "error"'),
    ("orchestrator", "Execute orchestrator commands", 'sentenial_cli_full.py orchestrator -a "update_policy" -p \'{"policy_id": "123"}\''),
    ("telemetry", "Stream real-time telemetry", 'sentenial_cli_full.py telemetry -s "network_monitor" -f "high_severity"'),
    ("alert", "Dispatch alerts", 'sentenial_cli_full.py alert -t "ransomware_detected" -s "high"'),
    ("simulate", "Run threat simulations", 'sentenial_cli_full.py simulate -sc "phishing_campaign"'),
]

def generate_pdf(output_path="SentenialX_CLI_CheatSheet.pdf"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Sentenial-X CLI Cheat Sheet")
    c.setFont("Helvetica", 12)
    y = height - 80

    for cmd, desc, example in CLI_COMMANDS:
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{cmd}")
        y -= 15
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"Description: {desc}")
        y -= 15
        c.drawString(60, y, f"Example: {example}")
        y -= 25

        if y < 80:
            c.showPage()
            y = height - 50

    c.save()
    print(f"Cheat sheet PDF generated at {output_path}")

if __name__ == "__main__":
    generate_pdf()
