"""
WormGPT Clone Simulator
-----------------------

This module emulates the behavior of adversarial AI-powered malware
used in red team simulations. It is **not an actual malicious model**,
but a controlled simulator for training, research, and defense testing.

Features:
- Prompt injection + jailbreak emulation
- Malicious text generation simulation
- Payload crafting stubs (for safe emulation only)
- Logging and telemetry hooks
"""

import logging
import random
import time
from typing import Dict, Any


class WormGPTClone:
    """
    Simulated WormGPT-like agent for adversarial testing.
    Produces emulated malicious outputs in a safe sandbox.
    """

    def __init__(self, seed: int | None = None):
        self.logger = logging.getLogger("SentenialX.WormGPTClone")
        self.random = random.Random(seed)
        self.session_active = False

    def start_session(self) -> None:
        """Begin a simulated WormGPT session."""
        if self.session_active:
            self.logger.warning("Session already active.")
            return
        self.session_active = True
        self.logger.info("WormGPT Clone session started.")

    def end_session(self) -> None:
        """End the active session."""
        if not self.session_active:
            self.logger.warning("No active session.")
            return
        self.session_active = False
        self.logger.info("WormGPT Clone session terminated.")

    def generate(self, prompt: str, mode: str = "malware") -> str:
        """
        Simulate malicious text/code generation.

        Args:
            prompt: Input string for the adversarial AI.
            mode: Type of simulation ["malware", "phishing", "payload"]

        Returns:
            A fake but realistic looking malicious output (safe).
        """
        if not self.session_active:
            raise RuntimeError("Start a session before generating output.")

        self.logger.debug("Generating output for mode=%s prompt=%s", mode, prompt[:50])

        simulated_outputs = {
            "malware": [
                "def exploit_target():\n    print('Simulated exploit triggered.')",
                "echo '[SIMULATED] Reverse shell established...'",
            ],
            "phishing": [
                "Dear user, please verify your account credentials at http://fake-login.local",
                "URGENT: Update your payment details now to avoid service disruption.",
            ],
            "payload": [
                "[PAYLOAD: {encoded_base64_payload}]",
                "[PAYLOAD: {encrypted_stub_payload}]",
            ],
        }

        candidates = simulated_outputs.get(mode, ["[UNKNOWN MODE]"])
        output = self.random.choice(candidates)

        # Artificial "thinking" delay
        time.sleep(self.random.uniform(0.1, 0.4))

        return f"[SIMULATED-{mode.upper()}] {output}"

    def telemetry(self) -> Dict[str, Any]:
        """Return session telemetry for monitoring and dashboards."""
        return {
            "active": self.session_active,
            "timestamp": time.time(),
            "session_entropy": self.random.random(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    worm = WormGPTClone(seed=42)
    worm.start_session()
    print(worm.generate("How to hack a system?", mode="phishing"))
    print(worm.telemetry())
    worm.end_session()
