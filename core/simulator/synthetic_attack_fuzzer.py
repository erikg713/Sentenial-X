"""
Synthetic Attack Fuzzer
-----------------------

Part of the Sentenial-X simulator subsystem.

This module generates *synthetic attack payloads* that mimic fuzzing attempts,
protocol abuses, and evasive sequences. It is safe for research/testing use
and does not contain any real malicious exploit code.

Features:
- Random payload generator (HTTP, SQLi-like, XSS-like, binary junk, etc.)
- Configurable fuzzing sessions
- Telemetry export for dashboards
- Logging integration
"""

import logging
import random
import string
import time
from typing import Dict, Any, List


class SyntheticAttackFuzzer:
    """Controlled synthetic attack fuzzer for safe red-team simulation."""

    PAYLOAD_TEMPLATES = {
        "http": [
            "GET /{junk} HTTP/1.1\r\nHost: example.local\r\n\r\n",
            "POST /login HTTP/1.1\r\nHost: example.local\r\nContent-Length: 20\r\n\r\nuser={junk}&pass=123",
        ],
        "sql_injection": [
            "' OR '1'='1 -- ",
            "'; DROP TABLE users; --",
            "admin'/*{junk}*/'--",
        ],
        "xss": [
            "<script>alert('{junk}')</script>",
            "<img src=x onerror=alert('{junk}')>",
        ],
        "binary": [
            "".join(chr(random.randint(0, 255)) for _ in range(16)),
            "".join(chr(random.randint(0, 255)) for _ in range(32)),
        ],
    }

    def __init__(self, seed: int | None = None):
        self.logger = logging.getLogger("SentenialX.SyntheticAttackFuzzer")
        self.random = random.Random(seed)
        self.active = False
        self.history: List[Dict[str, Any]] = []

    def start(self) -> None:
        """Begin fuzzing session."""
        if self.active:
            self.logger.warning("Fuzzer session already active.")
            return
        self.active = True
        self.logger.info("Synthetic Attack Fuzzer started.")

    def stop(self) -> None:
        """Stop fuzzing session."""
        if not self.active:
            self.logger.warning("No active fuzzer session.")
            return
        self.active = False
        self.logger.info("Synthetic Attack Fuzzer stopped.")

    def generate_payload(self, category: str = "http") -> str:
        """
        Generate a synthetic attack payload for the given category.

        Args:
            category: One of ["http", "sql_injection", "xss", "binary"]

        Returns:
            Synthetic payload string
        """
        if not self.active:
            raise RuntimeError("Start the fuzzer before generating payloads.")

        self.logger.debug("Generating payload for category=%s", category)

        template_list = self.PAYLOAD_TEMPLATES.get(category, [])
        if not template_list:
            return "[UNKNOWN CATEGORY]"

        template = self.random.choice(template_list)
        junk = "".join(self.random.choices(string.ascii_letters + string.digits, k=8))
        payload = template.replace("{junk}", junk)

        record = {
            "timestamp": time.time(),
            "category": category,
            "payload": payload,
        }
        self.history.append(record)

        return f"[FUZZ-{category.upper()}] {payload}"

    def fuzz_cycle(self, categories: List[str] | None = None, count: int = 5) -> List[str]:
        """
        Run a cycle of fuzzing payloads across categories.

        Args:
            categories: List of categories to fuzz (default: all)
            count: Number of payloads to generate

        Returns:
            List of generated payloads
        """
        if categories is None:
            categories = list(self.PAYLOAD_TEMPLATES.keys())

        outputs = []
        for _ in range(count):
            cat = self.random.choice(categories)
            outputs.append(self.generate_payload(cat))
            time.sleep(self.random.uniform(0.05, 0.2))  # mimic fuzz pacing
        return outputs

    def telemetry(self) -> Dict[str, Any]:
        """Export fuzzing session telemetry."""
        return {
            "active": self.active,
            "history_size": len(self.history),
            "last_payload": self.history[-1] if self.history else None,
            "timestamp": time.time(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    fuzzer = SyntheticAttackFuzzer(seed=1337)
    fuzzer.start()
    print(fuzzer.generate_payload("sql_injection"))
    print(fuzzer.fuzz_cycle(count=3))
    print(fuzzer.telemetry())
    fuzzer.stop()
