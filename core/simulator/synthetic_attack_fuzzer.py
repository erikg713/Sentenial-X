# core/simulator/synthetic_attack_fuzzer.py
# -*- coding: utf-8 -*-
"""
Synthetic Attack Fuzzer
-----------------------

Generates safe, synthetic fuzzing payloads for testing detection and parsing logic.
"""

from __future__ import annotations

import logging
import random
import string
import time
from typing import Any, Dict, List, Optional

from . import BaseSimulator

_logger = logging.getLogger("SentenialX.Simulator.SyntheticAttackFuzzer")
if not _logger.handlers:
    _logger.addHandler(logging.NullHandler())


class SyntheticAttackFuzzer(BaseSimulator):
    DEFAULT_TEMPLATES = {
        "http": [
            "GET /{junk} HTTP/1.1\r\nHost: example.local\r\n\r\n",
            "POST /login HTTP/1.1\r\nHost: example.local\r\nContent-Length: {len}\r\n\r\nuser={junk}&pass=123",
        ],
        "sql_injection": [
            "' OR '1'='1' -- ",
            "'; -- [SIMULATED]",
            "admin'/*{junk}*/'--",
        ],
        "xss": [
            "<script>console.log('{junk}')</script>",
            "<img src=x onerror=console.log('{junk}')>",
        ],
        "binary": [
            None,  # replaced programmatically
        ],
    }

    def __init__(self, seed: Optional[int] = None, name: str | None = None) -> None:
        super().__init__(name=name or "SyntheticAttackFuzzer")
        self.random = random.Random(seed)
        self.history: List[Dict[str, Any]] = []

    def _mk_junk(self, length: int = 12) -> str:
        return "".join(self.random.choices(string.ascii_letters + string.digits, k=length))

    def generate_payload(self, category: str = "http") -> str:
        tmpl_list = self.DEFAULT_TEMPLATES.get(category, [])
        if not tmpl_list:
            return "[UNKNOWN_CATEGORY]"

        tmpl = self.random.choice(tmpl_list)
        if tmpl is None:
            # generate binary-like safe placeholder
            payload = "".join(chr(self.random.randint(32, 126)) for _ in range(16))
        else:
            junk = self._mk_junk(10)
            payload = tmpl.format(junk=junk, len=len(junk) + 10)

        record = {"timestamp": time.time(), "category": category, "payload": payload}
        self.history.append(record)
        return payload

    def run(self, categories: List[str] | None = None, count: int = 5) -> Dict[str, Any]:
        """
        Generate `count` payloads across the given categories.

        Returns:
            structured dict with list of generated payloads and basic scoring.
        """
        if not self.active:
            raise RuntimeError("Simulator not started; call .start() first")

        if categories is None:
            categories = list(self.DEFAULT_TEMPLATES.keys())

        outputs: List[Dict[str, Any]] = []
        for _ in range(max(1, int(count))):
            cat = self.random.choice(categories)
            payload = self.generate_payload(cat)
            outputs.append({"category": cat, "payload_summary": payload[:400], "timestamp": time.time()})
            time.sleep(self.random.uniform(0.01, 0.05))  # small pacing

        severity = 1 + min(len(outputs) // 2, 8)
        return {
            "name": self.name,
            "timestamp": time.time(),
            "generated": outputs,
            "history_size": len(self.history),
            "severity": int(severity),
        }

    def telemetry(self) -> Dict[str, Any]:
        t = super().telemetry()
        t.update({"history_size": len(self.history)})
        return t
