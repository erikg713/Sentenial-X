"""
apps/dashboard/main.py
----------------------
Entry point for the Sentenial-X Dashboard application.
Initializes layout, frontend placeholder, and simulates live data updates.
"""

import time
from apps.dashboard.frontend_placeholder import FrontendPlaceholder
from apps.dashboard.layout import DashboardLayout
from api.routes import telemetry, orchestrator, cortex, wormgpt, exploits

def simulate_api_fetch():
    """Simulate fetching data from API endpoints for the dashboard."""
    telemetry_data = {"cpu": "28%", "memory": "63%", "disk": "120GB", "network": "200Mbps"}
    orchestrator_status = "running"
    cortex_analysis = {"threat": {"name": "malware_xyz"}, "confidence": 0.93}
    wormgpt_emulation = {"payload": {"command": "scan"}, "status": "emulated"}
    exploits_list = ["ms17-010", "struts_rce", "cve-2021-44228"]
    return telemetry_data, orchestrator_status, cortex_analysis, wormgpt_emulation, exploits_list


def main_loop(dashboard: DashboardLayout, interval: float = 5.0):
    """Main loop to update dashboard periodically."""
    try:
        while True:
            telemetry_data, orchestrator_status, cortex_analysis, wormgpt_emulation, exploits_list = simulate_api_fetch()
            
            dashboard.update_telemetry(telemetry_data)
            dashboard.update_orchestrator_status(orchestrator_status)
            dashboard.update_cortex_analysis(cortex_analysis)
            dashboard.update_wormgpt_emulation(wormgpt_emulation)
            dashboard.update_exploits_list(exploits_list)

            dashboard.refresh_layout()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[Dashboard] Exiting dashboard loop.")


if __name__ == "__main__":
    frontend = FrontendPlaceholder()
    dashboard = DashboardLayout(frontend=frontend)
    print("[Dashboard] Starting Sentenial-X Dashboard...")
    main_loop(dashboard)
