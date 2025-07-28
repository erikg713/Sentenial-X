import re

class SemanticAnalyzer:
    def __init__(self):
        self.patterns = {
            "malware": r"(suspicious|encoded|injection|obfuscation)",
            "breach": r"(unauthorized|escalation|root access|data leak)",
            "exploit": r"(rce|overflow|cve|buffer|memory corruption)"
        }

    def analyze(self, signal):
        text_blob = signal.get("description", "").lower()
        for label, pattern in self.patterns.items():
            if re.search(pattern, text_blob):
                return {"intent": label, "matched_pattern": pattern}
        return {"intent": "unknown", "matched_pattern": None}
