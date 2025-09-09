# -*- coding: utf-8 -*-
"""
run_all_envs.py
----------------

Full environment simulation and test runner for Sentenial-X.

- Sets up a virtual environment simulation
- Loads plugins
- Discovers simulators
- Executes AI core predictive models
- Runs all pytest tests
- Collects telemetry and logs outputs
"""

from __future__ import annotations
import subprocess
import sys
import logging
from pathlib import Path

from scripts.load_plugins import load_all_plugins
from core.simulator import discover_simulators, EmulationManager, TelemetryCollector
from ai_core.predictive_model import PredictiveModel

# ---------------------------
# Logging
# ---------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "env_simulation.log",
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger("SentenialX.EnvSim")

# ---------------------------
# Environment Simulation
# ---------------------------
def setup_environment() -> None:
    logger.info("Starting environment simulation...")

    # Step 1: Load plugins
    plugins = load_all_plugins()
    logger.info("Loaded %d plugins.", len(plugins))

    # Step 2: Discover simulators
    simulators = discover_simulators()
    logger.info("Discovered %d simulator(s).", len(simulators))

    # Step 3: Register simulators in EmulationManager
    manager = EmulationManager()
    for sim in simulators:
        try:
            manager.register(sim)
            logger.info("Registered simulator: %s", getattr(sim, "name", sim.__class__.__name__))
        except Exception as e:
            logger.error("Failed to register simulator %s: %s", sim, e)

    # Step 4: Initialize AI Core predictive model
    model = PredictiveModel()
    logger.info("Predictive model initialized: %s", model.__class__.__name__)

    # Step 5: Execute simulators and AI predictions
    for sim in simulators:
        try:
            sim_output = sim.run() if hasattr(sim, "run") else "no_run_method"
            prediction = model.predict(sim_output)
            logger.info("Simulator: %s | AI Prediction: %s", getattr(sim, "name", sim.__class__.__name__), prediction)
        except Exception as e:
            logger.error("Error during simulator execution or AI prediction: %s", e)

    # Step 6: Collect telemetry
    telemetry = TelemetryCollector()
    telemetry.collect(manager)
    logger.info("Telemetry collected: %s", telemetry.summary())

    logger.info("Environment simulation setup complete.")

# ---------------------------
# Run all pytest tests
# ---------------------------
def run_tests() -> None:
    logger.info("Running all pytest tests...")
    try:
        subprocess.run([sys.executable, "-m", "pytest", "tests"], check=True)
        logger.info("All tests completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error("Some tests failed: %s", e)

# ---------------------------
# Main Entrypoint
# ---------------------------
if __name__ == "__main__":
    setup_environment()
    run_tests()
