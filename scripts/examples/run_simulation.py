# -*- coding: utf-8 -*-
"""
Example: Run Sentenial-X Simulators + Telemetry + Attack Playbook
------------------------------------------------------------------

Demonstrates:
- Instantiating simulators (WormGPTClone, SyntheticAttackFuzzer, BlindSpotTracker)
- Running simulations sequentially via EmulationManager
- Collecting telemetry via TelemetryCollector
- Executing a sample AttackPlaybook
- Outputting unified structured results
"""

from __future__ import annotations

import json
import logging
import time

# Import simulator subsystem
from core.simulator import (
    discover_simulators,
    EmulationManager,
    TelemetryCollector,
)
from core.simulator.attack_playbook import create_playbook

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentenialX.RunSimulation")

# ---------------------------------------------------------------------------
# Step 1: Discover & register simulators
# ---------------------------------------------------------------------------
simulators = discover_simulators()
logger.info("Discovered %d simulators", len(simulators))

manager = EmulationManager(simulators=simulators)
manager.start_all()

# ---------------------------------------------------------------------------
# Step 2: Run all simulators sequentially
# ---------------------------------------------------------------------------
sim_results = manager.run_all(sequential=True)
logger.info("Simulator run completed. Collected results from %d simulators", len(sim_results))

# ---------------------------------------------------------------------------
# Step 3: Collect telemetry
# ---------------------------------------------------------------------------
collector = TelemetryCollector(history_limit=500)

for res in sim_results:
    # Add raw telemetry (simulator internal metrics)
    if "name" in res:
        sim_name = res["name"]
    else:
        sim_name = "unknown_sim"
    telemetry = {"simulator": sim_name, "result_summary": res, "timestamp": time.time()}
    collector.add(telemetry)

telemetry_report = collector.report()
logger.info("Telemetry collected: %d records", len(telemetry_report))
logger.info("Telemetry summary: %s", collector.summary())

# ---------------------------------------------------------------------------
# Step 4: Execute sample AttackPlaybook
# ---------------------------------------------------------------------------
playbook = create_playbook()
playbook_results = playbook.run()

logger.info("AttackPlaybook executed: %s steps", len(playbook.steps))

# ---------------------------------------------------------------------------
# Step 5: Output unified structured report
# ---------------------------------------------------------------------------
unified_report = {
    "simulators_run": [sim.name for sim in simulators],
    "simulator_results": sim_results,
    "telemetry_summary": collector.summary(),
    "playbook_id": playbook.id,
    "playbook_results": playbook_results,
    "timestamp": time.time(),
}

# Pretty-print JSON report
print(json.dumps(unified_report, indent=2))

# ---------------------------------------------------------------------------
# Step 6: Stop all simulators
# ---------------------------------------------------------------------------
manager.stop_all()
logger.info("All simulators stopped. Example run complete.")
