
"""
sentenial_core.interfaces.detection_engine

Core detection engine for Sentenial-X-A.I.
Implements modular detection strategies for scalable, adaptive threat analysis.

Author: Erik G. <your.email@domain.com>
"""

from typing import Any, Dict, List, Protocol, Optional
from datetime import datetime
import logging

logger = logging.getLogger("sentenial_core.detection")
logger.setLevel(logging.INFO)

class DetectionResult:
    """
    Standardized result object for detection operations.
    """
    def __init__(self, threat_type: str, details: Dict[str, Any], detected_at: Optional[datetime] = None):
        self.threat_type = threat_type
        self.details = details
        self.detected_at = detected_at or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat_type": self.threat_type,
            "details": self.details,
            "detected_at": self.detected_at.isoformat(),
        }

    def __repr__(self) -> str:
        return f"<DetectionResult threat_type={self.threat_type} detected_at={self.detected_at.isoformat()}>"

class DetectionStrategy(Protocol):
    """
    Protocol for detection strategies. Implement this interface for custom logic.
    """
    def detect(self, data: Dict[str, Any]) -> Optional[DetectionResult]:
        ...

class RuleBasedDetection:
    """
    Simple rule-based detection strategy.
    """
    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules

    def detect(self, data: Dict[str, Any]) -> Optional[DetectionResult]:
        for rule in self.rules:
            if all(data.get(k) == v for k, v in rule.get("match", {}).items()):
                logger.info(f"Rule match: {rule.get('name', 'unnamed')} for data: {data}")
                return DetectionResult(
                    threat_type=rule.get("threat_type", "unknown"),
                    details={"matched_rule": rule.get("name", "unnamed"), "data": data},
                )
        return None

class AnomalyDetection:
    """
    Placeholder for anomaly detection (statistical/ML-based).
    Expand with actual ML logic as needed.
    """
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def detect(self, data: Dict[str, Any]) -> Optional[DetectionResult]:
        # Placeholder: implement real anomaly logic here
        if "score" in data and data["score"] > self.threshold:
            logger.info("Anomaly detected with score: %s", data["score"])
            return DetectionResult(
                threat_type="anomaly",
                details={"score": data["score"], "data": data},
            )
        return None

class DetectionEngine:
    """
    Main detection engine, supporting multiple strategies.
    """
    def __init__(self):
        self.strategies: List[DetectionStrategy] = []

    def register_strategy(self, strategy: DetectionStrategy) -> None:
        self.strategies.append(strategy)
        logger.info(f"Registered detection strategy: {type(strategy).__name__}")

    def detect(self, data: Dict[str, Any]) -> List[DetectionResult]:
        results = []
        logger.debug(f"Starting detection for data: {data}")
        for strategy in self.strategies:
            result = strategy.detect(data)
            if result:
                logger.info(f"Detection result: {result}")
                results.append(result)
        if not results:
            logger.debug("No detections found.")
        return results

    def load_default_strategies(self) -> None:
        # Example: add some default rules and anomaly detection
        self.register_strategy(RuleBasedDetection([
            {"name": "admin_login_from_abroad", "match": {"event": "login", "role": "admin", "country": "RU"}, "threat_type": "suspicious_login"},
            {"name": "multiple_failed_logins", "match": {"event": "failed_login", "count": 5}, "threat_type": "brute_force"},
        ]))
        self.register_strategy(AnomalyDetection(threshold=0.9))

# Example usage (for integration/unit tests, not part of core module)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = DetectionEngine()
    engine.load_default_strategies()
    test_data = [
        {"event": "login", "role": "admin", "country": "RU"},
        {"event": "failed_login", "count": 5},
        {"event": "login", "score": 0.95},
        {"event": "user_action", "count": 1},
    ]
    for data in test_data:
        detections = engine.detect(data)
        for detection in detections:
            print(detection)
