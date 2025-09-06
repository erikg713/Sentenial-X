# -*- coding: utf-8 -*-
"""
Sentenial-X Simulator Package
-----------------------------

The simulator subsystem provides synthetic attack and defense emulators.
These are safe training/analysis tools that mimic behaviors of threats,
vulnerabilities, and detection gaps without executing real exploits.

Modules:
- wormgpt_clone:  Simulates adversarial AI text generation.
- synthetic_attack_fuzzer: Generates synthetic fuzzing payloads.
- blind_spot_tracker: Identifies defensive monitoring blind spots.

Common Interface:
- BaseSimulator: abstract helper class
- SimulatorProtocol: typing protocol for pluggable simulators
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict, Protocol, runtime_checkable

__all__ = [
    "SimulatorProtocol",
    "BaseSimulator",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SentenialX.Simulator")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Protocol + Base
# ---------------------------------------------------------------------------
@runtime_checkable
class SimulatorProtocol(Protocol):
    """Protocol all simulators should satisfy."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run(self, *args: Any, **kwargs: Any) -> Any: ...
    def telemetry(self) -> Dict[str, Any]: ...


class BaseSimulator(abc.ABC):
    """
    Base class for simulators in Sentenial-X.

    Provides:
    - Lifecycle (`start`, `stop`)
    - Abstract `run`
    - Default `telemetry` hook
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.started_at: float | None = None
        self.logger = logging.getLogger(f"SentenialX.{self.__class__.__name__}")

    def start(self) -> None:
        if self.active:
            self.logger.warning("Simulator already active.")
            return
        self.active = True
        self.started_at = time.time()
        self.logger.info("%s started.", self.__class__.__name__)

    def stop(self) -> None:
        if not self.active:
            self.logger.warning("Simulator not active.")
            return
        self.active = False
        self.logger.info("%s stopped.", self.__class__.__name__)

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute one simulation cycle."""
        raise NotImplementedError

    def telemetry(self) -> Dict[str, Any]:
        """Default telemetry payload (override if needed)."""
        return {
            "simulator": self.__class__.__name__,
            "active": self.active,
            "uptime": time.time() - self.started_at if self.active and self.started_at else 0,
            "timestamp": time.time(),
        }
