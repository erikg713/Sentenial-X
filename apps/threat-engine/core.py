# apps/threat-engine/core.py
from .llm_analyzer import LLMAnalyzer
from .telemetry_analyzer import TelemetryAnalyzer
from .rules_engine import RulesEngine

class ThreatEngine:
    """
    Core threat detection engine combining LLM, telemetry, and rules.
    """

    def __init__(self):
        self.llm = LLMAnalyzer()
        self.telemetry = TelemetryAnalyzer()
        self.rules = RulesEngine()

    def analyze(self, agent_logs, telemetry_data):
        """
        Analyze agent data and return detected threats.
        """
        threats = []
        # Rule-based threats
        threats.extend(self.rules.scan(agent_logs))
        # Telemetry anomalies
        threats.extend(self.telemetry.detect_anomalies(telemetry_data))
        # LLM-based threat predictions
        threats.extend(self.llm.predict(agent_logs, telemetry_data))
        return threats 