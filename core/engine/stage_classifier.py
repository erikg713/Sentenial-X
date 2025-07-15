"""
Classifies events into ATT&CK stages using heuristics.
"""

from typing import Dict, Any

def classify_stage(event: Dict[str, Any]) -> str:
    cmd = event.get("command", "").lower()

    if "ftp" in cmd or "curl" in cmd or "scp" in cmd:
        return "Exfiltration"
    if "net user" in cmd or "whoami" in cmd or "tasklist" in cmd:
        return "Reconnaissance"
    if "psexec" in cmd or "wmic" in cmd or "winrm" in cmd:
        return "Lateral Movement"
    if "schtasks" in cmd or "reg add" in cmd or "startup" in cmd:
        return "Persistence"
    if "cmd.exe" in cmd or "powershell" in cmd or "wscript" in cmd:
        return "Execution"
    if "vssadmin delete" in cmd or "amsi" in cmd:
        return "Defense Evasion"

    return "Unknown"

