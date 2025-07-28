import re
from sentenial_x.core.cortex.model.bert_classifier import BERTThreatClassifier

class SemanticAnalyzer:
    def __init__(self):
        self.classifier = BERTThreatClassifier()

    def analyze(self, signal: dict):
        description = signal.get("description", "")
        if not description:
            return {"intent": "unknown", "confidence": 0.0}

        intent, confidence = self.classifier.classify(description)
        return {
            "intent": intent,
            "confidence": confidence
        }

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
