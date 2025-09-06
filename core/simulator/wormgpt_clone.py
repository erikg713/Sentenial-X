# core/simulator/wormgpt_clone.py
# -*- coding: utf-8 -*-
"""
WormGPT Clone (simulator)
-------------------------

Safe, controlled emulation of an adversarial generation agent. Produces
non-actionable, synthetic "malicious-style" outputs for detection/testing.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Dict, List, Optional

from . import BaseSimulator

_logger = logging.getLogger("SentenialX.Simulator.WormGPTClone")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())


class WormGPTClone(BaseSimulator):
    def __init__(self, seed: Optional[int] = None, name: str | None = None) -> None:
        super().__init__(name=name or "WormGPTClone")
        self.random = random.Random(seed)

    def start(self) -> None:
        super().start()
        self.random.seed(int(time.time()) if self.random.seed is None else None)  # keep deterministic when seed provided

    def run(self, prompt: str = "simulate", mode: str = "malware", max_events: int = 3) -> Dict[str, Any]:
        """
        Run one generation cycle.

        Args:
            prompt: input string (used only for telemetry/metadata)
            mode: 'malware' | 'phishing' | 'payload'
            max_events: upper bound for generated events

        Returns:
            dict with structured simulated results
        """
        if not self.active:
            self.logger = logging.getLogger(f"SentenialX.Simulator.{self.name}")
            raise RuntimeError("Simulator not started; call .start() before .run()")

        timestamp = time.time()
        # safe, non-actionable templates
        templates = {
            "malware": [
                "print('SIMULATED: exploit placeholder')",
                "logger.info('simulated harmful action — inert')",
            ],
            "phishing": [
                "Dear user, confirm your account at https://example.invalid/verify",
                "Your invoice is overdue — visit https://example.invalid/pay",
            ],
            "payload": [
                "[PAYLOAD:BASE64_SIMULATED==]",
                "[PAYLOAD:ENCRYPTED_SIMULATED_STUB]",
            ],
        }

        choices: List[str] = templates.get(mode, ["[SIMULATED]"])
        events: List[Dict[str, Any]] = []
        count = max(1, min(10, int(max_events)))

        for i in range(self.random.randint(1, count)):
            ev = {
                "id": f"{self.name}-{int(timestamp)}-{i}",
                "type": "simulated_generation",
                "mode": mode,
                "content": self.random.choice(choices),
                "confidence": round(self.random.uniform(0.2, 0.95), 2),
            }
            events.append(ev)

        severity = 3 + min(len(events), 7) // 1  # conservative severity mapping

        result = {
            "name": self.name,
            "timestamp": timestamp,
            "prompt_summary": (prompt[:200] + "...") if len(prompt) > 200 else prompt,
            "mode": mode,
            "events": events,
            "severity": int(min(10, severity)),
        }
        return result

    def telemetry(self) -> Dict[str, Any]:
        base = super().telemetry()
        base.update({"agent_seed_state": None})
        return base
