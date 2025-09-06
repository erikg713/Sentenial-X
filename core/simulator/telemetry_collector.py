# -*- coding: utf-8 -*-
"""
core.simulator.telemetry_collector
----------------------------------

Collects telemetry from all simulators, maintains history, and provides structured reports.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger("SentenialX.Simulator.TelemetryCollector")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


class TelemetryCollector:
    """Centralized telemetry management for Sentenial-X simulators."""

    def __init__(self, history_limit: int = 100):
        self.history: deque[Dict[str, Any]] = deque(maxlen=history_limit)
        self.history_limit = history_limit

    def add(self, telemetry: Dict[str, Any]) -> None:
        """Add a new telemetry record."""
        telemetry["collected_at"] = time.time()
        self.history.append(telemetry)
        logger.debug("Telemetry added: %s", telemetry.get("simulator", "<unknown>"))

    def report(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Export telemetry history.

        Args:
            last_n: if specified, return only the last N records
        """
        if last_n is None:
            return list(self.history)
        return list(self.history)[-last_n:]

    def summary(self) -> Dict[str, Any]:
        """Return a summary of telemetry collected."""
        summary: Dict[str, Any] = {"total_records": len(self.history), "simulators": {}}
        for record in self.history:
            sim_name = record.get("simulator", "unknown")
            summary["simulators"].setdefault(sim_name, 0)
            summary["simulators"][sim_name] += 1
        return summary
