"""
Calculates stealth score for a given telemetry event.
"""

from typing import Dict, Any

def score_evasion(event: Dict[str, Any]) -> float:
    score = 0.0
    cmd = event.get("command", "").lower()

    if "bypass" in cmd:
        score += 0.5
    if "ep " in cmd or "-ep" in cmd:
        score += 0.4
    if "amsi" in cmd:
        score += 0.5
    if "hidden" in cmd or "windowstyle" in cmd:
        score += 0.3
    if "vssadmin delete" in cmd:
        score += 0.7

    return min(score, 1.0)

