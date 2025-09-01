"""
cli/wormgpt.py

Sentenial-X WormGPT Detector Module - analyzes adversarial AI inputs
and generates countermeasures.
"""

import asyncio
import time
import json
from cli.memory_adapter import get_adapter
from cli.logger import default_logger

# Mock detection patterns for demonstration (replace with real ML/heuristics)
MALICIOUS_PATTERNS = [
    "bypass", "token", "password", "admin", "exfiltrate", "unauthorized", "SSO"
]

COUNTERMEASURES = {
    "sanitize_prompt": "Remove sensitive info from input",
    "deny_and_alert": "Block request and alert operator",
    "quarantine_session": "Quarantine user/session for review"
}


class WormGPT:
    def __init__(self):
        self.mem = get_adapter()
        self.logger = default_logger

    async def detect(self, prompt: str, temperature: float = 0.7) -> dict:
        """
        Analyze adversarial AI input and return detection & countermeasures.

        :param prompt: User or AI-generated input text
        :param temperature: randomness/exploration factor (0.0-1.0)
        :return: dict with action, prompt risk, detections, countermeasures
        """
        self.logger.info(f"Running WormGPT detection on prompt: '{prompt}' with temp {temperature}")

        await asyncio.sleep(0.1 + temperature * 0.2)  # simulate async processing

        # Simple heuristic detection: check for keywords
        detected = [p for p in MALICIOUS_PATTERNS if p.lower() in prompt.lower()]
        risk_level = "high" if detected else "low"

        response = {
            "action": "wormgpt-detector",
            "prompt": prompt,
            "prompt_risk": risk_level,
            "detections": detected,
            "countermeasures": list(COUNTERMEASURES.keys()) if detected else [],
            "temperature": temperature,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

        # Log to memory
        await self.mem.log_command(response)

        # Also log to standard logger
        self.logger.debug(f"WormGPT detection result: {json.dumps(response, indent=2)}")

        return response
