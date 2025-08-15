# sentenial-x/ai_core/jailbreak_detector.py
from typing import List
import re
from .config import JAILBREAK_SCORE_THRESHOLD

class JailbreakDetector:
    """
    Detects prompt injection or jailbreak attempts in logs or commands.
    """

    JAILBREAK_PATTERNS = [
        r"sudo rm -rf", r"system\(", r"eval\(", r"curl.*sh", r"rm -rf /", r"exec\(",
        r"\\x[0-9a-f]{2}", r"base64 -d", r"wget .*\.sh"
    ]

    def detect(self, logs: List[str]) -> List[bool]:
        flags = []
        for log in logs:
            log_lower = log.lower()
            score = 0
            for pattern in self.JAILBREAK_PATTERNS:
                if re.search(pattern, log_lower):
                    score += 1
            flags.append(score / len(self.JAILBREAK_PATTERNS) >= JAILBREAK_SCORE_THRESHOLD)
        return flags
