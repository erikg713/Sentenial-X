"""
semantic_analyzer/evasion_scoring.py

Stealth & Evasion Scoring Module
--------------------------------
This module evaluates telemetry events for stealth/evasion techniques
commonly used by malware, red teams, and advanced persistent threats (APTs).

Features:
- Keyword heuristics for known evasion flags & commands
- Weighted scoring model (0.0 – 1.0)
- Context-based evaluation (process + command + arguments)
- Easy extension for ML-based scoring in future phases
"""

from typing import Dict, Any, List

# Known evasion indicators with weights
EVASION_KEYWORDS = {
    "bypass": 0.5,               # e.g., AMSI or UAC bypass
    "-ep": 0.4,                  # PowerShell execution policy bypass
    " ep ": 0.4,                 # Variant spacing
    "amsi": 0.5,                 # Anti-Malware Scan Interface tampering
    "hidden": 0.3,               # Hidden execution
    "windowstyle hidden": 0.4,   # PowerShell hidden window
    "vssadmin delete": 0.7,      # Shadow copy deletion
    "reg add": 0.3,              # Registry persistence/evasion
    "schtasks": 0.4,             # Scheduled task abuse
    "wmic process call create": 0.6,  # WMI process spawn (stealthy)
}


def score_evasion(event: Dict[str, Any]) -> float:
    """
    Calculates stealth/evasion score for a given telemetry event.

    Args:
        event (Dict[str, Any]): Telemetry data with keys such as:
            - command: str (command line string)
            - process: str (process name)
            - args: List[str] (parsed arguments)

    Returns:
        float: Score between 0.0 (benign) and 1.0 (highly evasive)
    """
    score = 0.0
    cmd = event.get("command", "").lower()
    process = event.get("process", "").lower()
    args: List[str] = event.get("args", [])

    # Keyword-based scoring
    for keyword, weight in EVASION_KEYWORDS.items():
        if keyword in cmd:
            score += weight

    # Context-based scoring
    if process in ["powershell.exe", "pwsh.exe"]:
        score += 0.2  # suspicious context
    if "encodedcommand" in cmd:
        score += 0.5  # base64 encoded commands
    if any(arg.startswith("-nop") for arg in args):
        score += 0.3  # No profile, stealthy launch

    # Normalize
    return min(score, 1.0)


if __name__ == "__main__":
    # Example test
    test_event = {
        "command": 'powershell.exe -nop -ep bypass -encodedcommand ...',
        "process": "powershell.exe",
        "args": ["-nop", "-ep", "bypass", "-encodedcommand", "ABCD1234=="]
    }
    print("Evasion Score:", score_evasion(test_event))
