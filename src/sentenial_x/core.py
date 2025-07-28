# src/sentenial_x/core.py
from sentenial_x.plugins import PluginBase
from sentenial_x.ai.pipeline import SessionAnalyzer

class AICorePlugin(PluginBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiate once on startup
        self.analyzer = SessionAnalyzer(self.config.get("model_name", "gpt-j-6B"))

    def on_traffic_analyzed(self, event):
        session = event.payload  # assume a Session object
        ai_report = self.analyzer.analyze(session)

        # Emit a new alert with AI’s findings
        self.emit("on_alert_raised", {
            "session_id": session.id,
            "findings": ai_report["findings"],
            "risk_score": ai_report["risk_score"],
            "remediation": ai_report["remediation"]
        })

