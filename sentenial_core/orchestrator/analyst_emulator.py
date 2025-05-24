import random
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AnalystEmulator:
    def __init__(self, max_thinking_time_sec: int = 5):
        self.max_thinking_time_sec = max_thinking_time_sec

    def emulate_analysis(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Emulates the decision-making of a human SOC analyst,
        based on alert severity, historical context, and attack patterns.
        """
        logger.debug(f"Emulating analysis for alert: {alert}")

        # Emulate analysis delay to simulate human reasoning
        thinking_time = random.uniform(1, self.max_thinking_time_sec)
        time.sleep(thinking_time)

        severity_score = self._calculate_severity(alert)
        recommended_action = self._recommend_action(severity_score)

        decision = {
            "alert_id": alert.get("alert_id", "unknown"),
            "severity_score": severity_score,
            "recommended_action": recommended_action,
            "analysis_time_sec": round(thinking_time, 2),
            "analyst_confidence": round(random.uniform(0.75, 0.99), 2),
            "comments": self._generate_comments(severity_score)
        }

        logger.debug(f"Analysis result: {decision}")
        return decision

    def _calculate_severity(self, alert: Dict[str, Any]) -> float:
        """
        Calculates severity score from alert attributes such as vector type,
        detected exploit complexity, and affected assets.
        Returns a float 0.0 (low) to 1.0 (critical).
        """
        vector = alert.get("vector", "").lower()
        complexity = alert.get("complexity", 0.5)  # default moderate complexity
        affected_assets = alert.get("affected_assets", 1)

        base_severity = 0.3  # base low severity

        # Increase severity based on vector
        if "zero-day" in vector or "ransomware" in vector:
            base_severity += 0.5
        elif "sql injection" in vector:
            base_severity += 0.4
        elif "xss" in vector:
            base_severity += 0.25
        elif "recon" in vector or "scan" in vector:
            base_severity += 0.1

        # Factor in complexity and asset impact
        severity = min(1.0, base_severity + complexity * 0.3 + 0.05 * affected_assets)
        return severity

    def _recommend_action(self, severity_score: float) -> str:
        if severity_score > 0.85:
            return "immediate_block_and_isolate"
        elif severity_score > 0.6:
            return "block_and_alert"
        elif severity_score > 0.3:
            return "monitor_and_log"
        else:
            return "passive_monitoring"

    def _generate_comments(self, severity_score: float) -> str:
        if severity_score > 0.85:
            return "Critical threat detected. Immediate action required."
        elif severity_score > 0.6:
            return "High severity. Block and notify SOC."
        elif severity_score > 0.3:
            return "Medium severity. Monitor closely."
        else:
            return "Low severity. Continue passive observation."

# Example Usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    emulator = AnalystEmulator()
    alert_example = {
        "alert_id": "abc123",
        "vector": "SQL Injection attempt",
        "complexity": 0.8,
        "affected_assets": 3
    }
    result = emulator.emulate_analysis(alert_example)
    print(result)
