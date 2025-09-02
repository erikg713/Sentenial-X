"""
Sentenial-X AI Core: Threat Classifier
--------------------------------------
Classifies and scores threats based on incoming payloads.
Integrates with the orchestrator, telemetry, and AI models.

Author: Sentenial-X Development Team
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from api.utils.logger import init_logger
from ai_core.model_loader import ModelLoader

logger = init_logger("ai_core.threat_classifier")


class ThreatClassifier:
    """
    ThreatClassifier evaluates incoming threat data and assigns severity scores.
    """

    _SEVERITY_MAP = {
        "critical": 90,
        "high": 75,
        "medium": 50,
        "low": 25,
        "info": 10,
    }

    def __init__(self):
        self.model_loader = ModelLoader()
        self.classification_log: List[Dict[str, Any]] = []
        logger.info("ThreatClassifier initialized")

    def classify_threat(self, threat_data: Dict[str, Any], model_name: str = "cortex_ai") -> Dict[str, Any]:
        """
        Classify a single threat payload using a specified model.
        """
        logger.debug("Classifying threat with model %s: %s", model_name, threat_data)
        try:
            model = self.model_loader.load_model(model_name)
            # Replace this with actual model inference logic
            severity = self._mock_classify(threat_data)
            score = self._SEVERITY_MAP.get(severity, 0)
            result = {
                "threat": threat_data,
                "severity": severity,
                "risk_score": score,
                "model": model["model_path"],
                "timestamp": datetime.utcnow().isoformat(),
            }
            self.classification_log.append(result)
            logger.info("Threat classified: %s", result)
            return result
        except Exception as e:
            logger.exception("Failed to classify threat: %s", e)
            return {
                "threat": threat_data,
                "severity": "unknown",
                "risk_score": 0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    def _mock_classify(self, threat_data: Dict[str, Any]) -> str:
        """
        Mock classification logic. Replace with actual AI inference.
        """
        indicators = str(threat_data).lower()
        if any(x in indicators for x in ["rce", "sql_injection", "priv_esc"]):
            return "critical"
        elif any(x in indicators for x in ["xss", "malware", "trojan"]):
            return "high"
        elif "suspicious" in indicators:
            return "medium"
        return "low"

    def get_classification_log(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all classified threats.
        """
        return list(self.classification_log)

    def clear_log(self):
        """
        Clears the classification log.
        """
        logger.warning("Clearing threat classification log with %d entries", len(self.classification_log))
        self.classification_log.clear()


# ------------------------
# CLI / Test Example
# ------------------------
if __name__ == "__main__":
    classifier = ThreatClassifier()
    sample_threats = [
        {"source": "192.168.1.10", "type": "rce", "payload": "os.system('id')"},
        {"source": "192.168.1.20", "type": "scan", "payload": "nmap scan"},
        {"source": "192.168.1.30", "type": "xss", "payload": "<script>alert(1)</script>"},
    ]

    for t in sample_threats:
        result = classifier.classify_threat(t)
        print(result)

    print("Classification log:", classifier.get_classification_log())
