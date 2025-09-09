# -*- coding: utf-8 -*-
"""
scripts.run_pipeline
-------------------

Executes the Sentenial-X emulation pipeline.

- Loads all plugins
- Discovers simulators
- Executes registered playbooks
- Collects telemetry
- Supports sequential or parallel execution
"""

from __future__ import annotations
import logging
from typing import Any, List

from core.simulator import EmulationManager, TelemetryCollector, discover_simulators
from scripts.load_plugins import load_all_plugins

# Configure logging
logger = logging.getLogger("SentenialX.Pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def run_pipeline(sequential: bool = True) -> None:
    """
    Run the main Sentenial-X emulation pipeline.

    Args:
        sequential (bool): Run simulators one by one if True, or concurrently if False.
    """
    logger.info("Starting Sentenial-X emulation pipeline...")

    # Step 1: Load plugins
    loaded_plugins = load_all_plugins()
    logger.info("Loaded %d plugins.", len(loaded_plugins))

    # Step 2: Discover simulators
    simulators: List[Any] = discover_simulators()
    if not simulators:
        logger.warning("No simulators found. Exiting pipeline.")
        return
    logger.info("Discovered %d simulator(s).", len(simulators))

    # Step 3: Initialize EmulationManager
    manager = EmulationManager()
    for sim in simulators:
        try:
            manager.register(sim)
            logger.info("Registered simulator: %s", getattr(sim, "name", sim.__class__.__name__))
        except Exception as e:
            logger.error("Failed to register simulator %s: %s", sim, e)

    # Step 4: Execute pipeline
    try:
        manager.run_all(sequential=sequential)
        logger.info("Pipeline execution completed successfully.")
    except Exception as e:
        logger.exception("Error during pipeline execution: %s", e)

    # Step 5: Collect telemetry
    try:
        telemetry_collector = TelemetryCollector()
        telemetry_collector.collect(manager)
        telemetry_summary = telemetry_collector.summary()
        logger.info("Telemetry summary: %s", telemetry_summary)
    except Exception as e:
        logger.error("Failed to collect telemetry: %s", e)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Sentenial-X emulation pipeline.")
    parser.add_argument("--sequential", action="store_true", help="Run simulators sequentially")
    args = parser.parse_args()

    run_pipeline(sequential=args.sequential)
