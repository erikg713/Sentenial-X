"""
Sentenial X :: Semantic Behavior Analyzer

This module performs semantic analysis on observed behaviors or extracted content,
classifying their intent (e.g., reconnaissance, exfiltration, persistence, evasion).

Techniques:
- Keyword & opcode pattern analysis
- Intent classification
- Command structure parsing
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("SemanticsAnalyzer")
logging.basicConfig(level=logging.INFO)

INTENT_KEYWORDS = {
    "reconnaissance": [
        r"ipconfig", r"whoami", r"hostname", r"nltest", r"net\s+view", r"nslookup",
        r"systeminfo", r"tasklist"
    ],
    "exfiltration": [
        r"ftp\s+", r"curl\s+-T", r"scp\s+", r"powershell.*Invoke-WebRequest.*-Method\s+POST",
        r"nc\s+-w"
    ],
    "persistence": [
        r"reg\s+add", r"schtasks\s+/create", r"powershell.*Set-ItemProperty",
        r"copy\s+.*\s+Startup", r"WMIC\s+startup"
    ],
    "evasion": [
        r"vssadmin\s+delete", r"powershell.*-ep\b", r"bypass", r"AMSI\b",
        r"Set-MpPreference", r"disable", r"rundll32\s+"
    ],
    "lateral_movement": [
        r"psexec", r"wmic.*process", r"at\s+", r"smbexec", r"Invoke-Command", r"winrm"
    ],
    "execution": [
        r"cmd\.exe", r"powershell", r"rundll32", r"mshta", r"wscript", r"cscript"
    ]
}


class SemanticsAnalyzer:
    def __init__(self, keywords_map: Optional[Dict[str, List[str]]] = None):
        """
        :param keywords_map: Optional override for intent keyword mapping
        """
        self.keywords = keywords_map or INTENT_KEYWORDS

    def analyze_command(self, command: str) -> Dict[str, Any]:
        """
        Analyzes a command or script line and infers semantic intent(s).
        :param command: Raw shell, PowerShell, or script command
        :return: Dictionary of intents and matched patterns
        """
        command_lower = command.lower()
        detected_intents = {}
        for intent, patterns in self.keywords.items():
            for pattern in patterns:
                if re.search(pattern, command_lower):
                    detected_intents.setdefault(intent, []).append(pattern)

        logger.debug(f"Analyzed command: {command}")
        return {
            "command": command,
            "intents": list(detected_intents.keys()),
            "matches": detected_intents
        }

    def analyze_sequence(self, commands: List[str]) -> Dict[str, Any]:
        """
        Analyze a sequence of commands to extract overall behavioral summary.
        :param commands: List of command lines or script blocks
        :return: Summary with dominant intent(s)
        """
        intent_counter = {}

        for cmd in commands:
            result = self.analyze_command(cmd)
            for intent in result["intents"]:
                intent_counter[intent] = intent_counter.get(intent, 0) + 1

        dominant_intents = sorted(intent_counter, key=intent_counter.get, reverse=True)
        return {
            "summary": dominant_intents[:3],
            "intent_counts": intent_counter
        }

