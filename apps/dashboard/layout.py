"""
apps/dashboard/layout.py
-----------------------
Defines layout and widget structure for the Sentenial-X Dashboard.
Supports telemetry, orchestrator, Cortex, WormGPT, and exploits sections.
"""

from typing import List, Dict, Any
from apps.dashboard.frontend_placeholder import FrontendPlaceholder

class DashboardLayout:
    """Manages the layout and widget updates for the dashboard."""

    def __init__(self, frontend: FrontendPlaceholder):
        self.frontend = frontend
        self.widgets: Dict[str, Dict[str, Any]] = {
            "telemetry": {},
            "orchestrator": {},
            "cortex": {},
            "wormgpt": {},
            "exploits": []
        }
        print("[DashboardLayout] Initialized dashboard layout")

    # ------------------------
    # Widget Update Methods
    # ------------------------
    def update_telemetry(self, data: Dict[str, Any]):
        self.widgets["telemetry"] = data
        self.frontend.render_telemetry(data)

    def update_orchestrator_status(self, status: str):
        self.widgets["orchestrator"] = {"status": status}
        self.frontend.render_orchestrator_status(status)

    def update_cortex_analysis(self, analysis: Dict[str, Any]):
        self.widgets["cortex"] = analysis
        self.frontend.render_cortex_analysis(analysis)

    def update_wormgpt_emulation(self, emulation: Dict[str, Any]):
        self.widgets["wormgpt"] = emulation
        self.frontend.render_wormgpt_emulation(emulation)

    def update_exploits_list(self, exploits: List[str]):
        self.widgets["exploits"] = exploits
        self.frontend.render_exploits_list(exploits)

    # ------------------------
    # Layout Control Methods
    # ------------------------
    def refresh_layout(self):
        """Refresh all widgets using current data."""
        print("[DashboardLayout] Refreshing dashboard layout")
        self.frontend.refresh()
        if self.widgets["telemetry"]:
            self.frontend.render_telemetry(self.widgets["telemetry"])
        if self.widgets["orchestrator"]:
            self.frontend.render_orchestrator_status(self.widgets["orchestrator"].get("status", "unknown"))
        if self.widgets["cortex"]:
            self.frontend.render_cortex_analysis(self.widgets["cortex"])
        if self.widgets["wormgpt"]:
            self.frontend.render_wormgpt_emulation(self.widgets["wormgpt"])
        if self.widgets["exploits"]:
            self.frontend.render_exploits_list(self.widgets["exploits"])


# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    frontend = FrontendPlaceholder()
    dashboard = DashboardLayout(frontend=frontend)

    # Simulate data update
    dashboard.update_telemetry({"cpu": "30%", "memory": "55%", "disk": "120GB"})
    dashboard.update_orchestrator_status("running")
    dashboard.update_cortex_analysis({"threat": {"name": "malware_xyz"}, "confidence": 0.95})
    dashboard.update_wormgpt_emulation({"payload": {"command": "scan"}, "status": "emulated"})
    dashboard.update_exploits_list(["ms17-010", "struts_rce", "cve-2021-44228"])

    # Refresh dashboard layout
    dashboard.refresh_layout()
