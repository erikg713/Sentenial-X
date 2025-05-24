
Directory: sentenial_core/cortex/intent_reconstructor.py

import re from typing import Dict, Any

class IntentReconstructor: """ Reconstructs attacker intent by analyzing semantic meaning of HTTP payloads. """ def init(self): self.patterns = [ (re.compile(r"(?i)(union.select)"), "SQL Injection"), (re.compile(r"(?i)<script.?>"), "XSS Attempt"), (re.compile(r"(?i)cmd=|/bin/bash"), "Command Injection"), ]

def analyze_payload(self, payload: str) -> Dict[str, Any]:
    for pattern, attack_type in self.patterns:
        if pattern.search(payload):
            return {
                "attack_type": attack_type,
                "payload": payload,
                "confidence": 0.95
            }
    return {
        "attack_type": "Unknown",
        "payload": payload,
        "confidence": 0.30
    }

