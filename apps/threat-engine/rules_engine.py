# apps/threat-engine/rules_engine.py
class RulesEngine:
    """
    Signature & heuristic based threat detection.
    """

    def __init__(self, signatures=None):
        # Example: list of regex patterns or known malicious strings
        self.signatures = signatures or ["malware", "exploit", "ransomware"]

    def scan(self, logs):
        """
        Scan logs for known malicious patterns.
        """
        threats = []
        for log in logs:
            for sig in self.signatures:
                if sig.lower() in str(log).lower():
                    threats.append({"type": "signature_match", "pattern": sig, "log": log})
        return threats