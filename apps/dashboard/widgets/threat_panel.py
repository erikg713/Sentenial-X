# apps/dashboard/widgets/threat_panel.py
class ThreatPanelWidget:
    def __init__(self):
        self.threats = []

    def add_threat(self, agent_id, level, desc):
        self.threats.append({"agent_id": agent_id, "level": level, "description": desc})

    def render(self):
        summary = {"low":0,"medium":0,"high":0}
        for t in self.threats:
            summary[t["level"]] += 1
        return summary
