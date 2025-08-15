# sentenial-x/ai_core/orchestrator.py
from .threat_classifier import ThreatClassifier
from .encoder import ThreatTextEncoder
from .jailbreak_detector import JailbreakDetector
from .countermeasure_predictor import CountermeasurePredictor

class AICoreOrchestrator:
    """
    Routes AI tasks in Sentenial-X:
    - Log encoding
    - Threat detection
    - Jailbreak detection
    - Countermeasure prediction
    """

    def __init__(self):
        self.encoder = ThreatTextEncoder()
        self.threat_model = ThreatClassifier()
        self.jailbreak_detector = JailbreakDetector()
        self.countermeasure_predictor = CountermeasurePredictor()

    def analyze_logs(self, logs):
        embeddings = self.encoder.encode(logs)
        threat_labels, threat_scores = self.threat_model.predict(logs)
        jailbreak_flags = self.jailbreak_detector.detect(logs)
        countermeasures = self.countermeasure_predictor.predict(logs, threat_scores)
        return {
            "embeddings": embeddings,
            "threat_labels": threat_labels,
            "threat_scores": threat_scores,
            "jailbreak_flags": jailbreak_flags,
            "countermeasures": countermeasures,
        }
