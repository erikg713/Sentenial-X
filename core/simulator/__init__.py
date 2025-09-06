# core/simulator/__init__.py
# -*- coding: utf-8 -*-
"""
Sentenial-X Simulator Package
-----------------------------

Unified simulator subsystem for Sentenial-X.

Public API:
- SimulatorProtocol
- BaseSimulator
- discover_simulators(): convenience to load default simulator instances
"""

from __future__ import annotations

import abc
import logging
import time
from typing import Any, Dict, Iterable, List, Protocol, runtime_checkable

logger = logging.getLogger("SentenialX.Simulator")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__all__ = [
    "SimulatorProtocol",
    "BaseSimulator",
    "discover_simulators",
]

@runtime_checkable
class SimulatorProtocol(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def run(self, *args: Any, **kwargs: Any) -> Any: ...
    def telemetry(self) -> Dict[str, Any]: ...

class BaseSimulator(abc.ABC):
    """
    Base class for all simulators.

    Subclasses should implement `run(...)` which performs one simulation cycle.
    """
    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self.active: bool = False
        self.started_at: float | None = None
        self.logger = logging.getLogger(f"SentenialX.Simulator.{self.name}")

    def start(self) -> None:
        if self.active:
            self.logger.debug("start() called but simulator already active.")
            return
        self.active = True
        self.started_at = time.time()
        self.logger.info("%s started.", self.name)

    def stop(self) -> None:
        if not self.active:
            self.logger.debug("stop() called but simulator not active.")
            return
        self.active = False
        self.logger.info("%s stopped.", self.name)

    @abc.abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute one simulation cycle and return a structured result dict.

        Return keys (recommended):
        - name: simulator name
        - timestamp: epoch seconds
        - result: simulator specific payload
        - severity: optional int 0-10
        """
        raise NotImplementedError

    def telemetry(self) -> Dict[str, Any]:
        return {
            "simulator": self.name,
            "active": self.active,
            "uptime_seconds": (time.time() - self.started_at) if (self.active and self.started_at) else 0.0,
            "timestamp": time.time(),
        }

def discover_simulators() -> List[BaseSimulator]:
    """
    Convenience: return one instance of each built-in simulator.

    This function imports submodules lazily to avoid heavy imports on package load.
    """
    sims: List[BaseSimulator] = []
    try:
        from .wormgpt_clone import WormGPTClone
        from .synthetic_attack_fuzzer import SyntheticAttackFuzzer
        from .blind_spot_tracker import BlindSpotTracker

        sims.append(WormGPTClone())
        sims.append(SyntheticAttackFuzzer())
        sims.append(BlindSpotTracker())
    except Exception as exc:
        logger.exception("Failed to discover simulators: %s", exc)
    return sims    logger.addHandler(logging.NullHandler())

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
