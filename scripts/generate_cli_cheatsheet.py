# -*- coding: utf-8 -*-
"""
Generate a CLI cheatsheet for Sentenial-X
-----------------------------------------

- Lists all available simulators
- Lists playbooks
- Lists EmulationManager actions
- Outputs formatted CLI-friendly cheatsheet
- Optionally generates a PDF version
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from core.simulator import discover_simulators
from core.simulator.attack_playbook import create_playbook

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentenialX.CLI.Cheatsheet")

# ---------------------------------------------------------------------------
# CLI Commands Examples
# ---------------------------------------------------------------------------
CLI_COMMANDS = [
    ("wormgpt-detector", "Analyze adversarial AI inputs", 'sentenial_cli_full.py wormgpt-detector -p "malicious prompt"'),
    ("blindspots", "Scan for detection blind spots", "sentenial_cli_full.py blindspots"),
    ("cortex", "Run NLP-based threat analysis", 'sentenial_cli_full.py cortex -s "/var/log/syslog" -f "error"'),
    ("orchestrator", "Execute orchestrator commands", 'sentenial_cli_full.py orchestrator -a "update_policy" -p \'{"policy_id": "123"}\''),
    ("telemetry", "Stream real-time telemetry", 'sentenial_cli_full.py telemetry -s "network_monitor" -f "high_severity"'),
    ("alert", "Dispatch alerts", 'sentenial_cli_full.py alert -t "ransomware_detected" -s "high"'),
    ("simulate", "Run threat simulations", 'sentenial_cli_full.py simulate -sc "phishing_campaign"'),
]

# ---------------------------------------------------------------------------
# Cheatsheet Generation
# ---------------------------------------------------------------------------
def generate_cheatsheet() -> Dict[str, Any]:
    """Generate structured cheatsheet information."""
    cheatsheet: Dict[str, Any] = {}

    # Discover simulators
    simulators = discover_simulators()
    cheatsheet["simulators"] = [sim.name for sim in simulators]

    # Sample playbook
    playbook = create_playbook()
    cheatsheet["playbooks"] = [
        {"id": playbook.id, "name": playbook.name, "steps": [s.id for s in playbook.steps]}
    ]

    # EmulationManager actions
    cheatsheet["emulation_manager"] = [
        "register(simulator)",
        "start_all()",
        "stop_all()",
        "run_all(sequential=True)",
        "collect_telemetry()",
    ]

    # TelemetryCollector actions
    cheatsheet["telemetry_collector"] = ["add(telemetry)", "report(last_n=None)", "summary()"]

    # CLI commands
    cheatsheet["cli_commands"] = [{"command": cmd, "description": desc, "example": ex} for cmd, desc, ex in CLI_COMMANDS]

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

    print("\nCLI Commands:")
    for cmd in cheatsheet.get("cli_commands", []):
        print(f"  - {cmd['command']}: {cmd['description']} (Example: {cmd['example']})")

    print("\n===================================\n")


# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------
def generate_pdf(output_path: str = "SentenialX_CLI_CheatSheet.pdf") -> None:
    """Generate a PDF version of the CLI cheatsheet."""
    cheatsheet = generate_cheatsheet()
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Header
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "Sentenial-X CLI Cheat Sheet")
    c.setFont("Helvetica", 12)
    y = height - 80

    # Write each CLI command
    for cmd in cheatsheet.get("cli_commands", []):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{cmd['command']}")
        y -= 15
        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"Description: {cmd['description']}")
        y -= 15
        c.drawString(60, y, f"Example: {cmd['example']}")
        y -= 25

        if y < 80:
            c.showPage()
            y = height - 50

    c.save()
    logger.info("Cheat sheet PDF generated at %s", output_path)
    print(f"✅ Cheat sheet PDF generated at {output_path}")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main(output_json: bool = False, pdf: bool = False) -> None:
    cheatsheet = generate_cheatsheet()
    if output_json:
        print(json.dumps(cheatsheet, indent=2))
    elif pdf:
        generate_pdf()
    else:
        print_cheatsheet(cheatsheet)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a CLI cheatsheet for Sentenial-X.")
    parser.add_argument("--json", action="store_true", help="Output cheatsheet as JSON")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF version of cheatsheet")
    args = parser.parse_args()

    main(output_json=args.json, pdf=args.pdf)
