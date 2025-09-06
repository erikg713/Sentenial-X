# -*- coding: utf-8 -*-
"""
core.simulator.emulation_manager
--------------------------------

Central orchestrator for Sentenial-X simulation engines.

Features:
- Manage lifecycle of multiple simulators
- Run simulators sequentially or threaded
- Aggregate structured results for dashboard or analysis
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional, Union

from . import BaseSimulator

logger = logging.getLogger("SentenialX.Simulator.EmulationManager")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class EmulationManager:
    """Orchestrates multiple simulator instances."""

    def __init__(self, simulators: Optional[List[BaseSimulator]] = None):
        self.simulators: List[BaseSimulator] = simulators or []
        self.active = False
        self.results: List[Dict[str, Any]] = []

    def register(self, simulator: BaseSimulator) -> None:
        """Add a simulator to the manager."""
        if simulator in self.simulators:
            logger.warning("Simulator %s already registered", simulator.name)
            return
        self.simulators.append(simulator)
        logger.info("Registered simulator: %s", simulator.name)

    def start_all(self) -> None:
        """Start all registered simulators."""
        if self.active:
            logger.warning("EmulationManager already active")
            return
        self.active = True
        for sim in self.simulators:
            sim.start()
        logger.info("All simulators started.")

    def stop_all(self) -> None:
        """Stop all simulators."""
        if not self.active:
            logger.warning("EmulationManager not active")
            return
        for sim in self.simulators:
            sim.stop()
        self.active = False
        logger.info("All simulators stopped.")

    def run_all(self, sequential: bool = True, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Run all simulators.

        Args:
            sequential: if True, run one after another; else run threaded
            kwargs: optional arguments passed to each simulator's `run`

        Returns:
            List of simulator results
        """
        self.results.clear()

        if sequential:
            for sim in self.simulators:
                try:
                    self.results.append(sim.run(**kwargs))
                except Exception as exc:
                    logger.exception("Simulator %s failed: %s", sim.name, exc)
        else:
            threads: List[threading.Thread] = []

            def worker(sim: BaseSimulator):
                try:
                    res = sim.run(**kwargs)
                    self.results.append(res)
                except Exception as exc:
                    logger.exception("Simulator %s failed: %s", sim.name, exc)

            for sim in self.simulators:
                t = threading.Thread(target=worker, args=(sim,))
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        return self.results

    def collect_telemetry(self) -> List[Dict[str, Any]]:
        """Return telemetry from all simulators."""
        telemetry_list: List[Dict[str, Any]] = []
        for sim in self.simulators:
            try:
                telemetry_list.append(sim.telemetry())
            except Exception as exc:
                logger.exception("Failed to collect telemetry from %s: %s", sim.name, exc)
        return telemetry_list
