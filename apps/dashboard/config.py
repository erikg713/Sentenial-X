"""
apps/dashboard/config.py
------------------------
Configuration for the Dashboard module of Sentenial-X.

Purpose:
- Centralize dashboard-specific settings.
- Manage widget configurations, refresh intervals, and display options.
- Integrate with API or AI modules as needed.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if .env exists
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR.parent / ".env")


class DashboardSettings:
    """Dashboard configuration settings."""

    # Refresh interval for real-time widgets (seconds)
    WIDGET_REFRESH_INTERVAL: int = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", 5))

    # Enable or disable debug mode for dashboard logs
    DEBUG: bool = os.getenv("DASHBOARD_DEBUG", "False").lower() in ("true", "1")

    # Maximum number of events shown per widget
    MAX_WIDGET_EVENTS: int = int(os.getenv("MAX_WIDGET_EVENTS", 50))

    # Theme settings (light/dark)
    THEME: str = os.getenv("DASHBOARD_THEME", "dark")

    # Optional integration endpoints
    TELEMETRY_API: str = os.getenv("TELEMETRY_API", "http://localhost:8000/api/telemetry")
    ORCHESTRATOR_API: str = os.getenv("ORCHESTRATOR_API", "http://localhost:8000/api/orchestrator")
    CORTEX_API: str = os.getenv("CORTEX_API", "http://localhost:8000/api/cortex")
    WORMGPT_API: str = os.getenv("WORMGPT_API", "http://localhost:8000/api/wormgpt")
    EXPLOITS_API: str = os.getenv("EXPLOITS_API", "http://localhost:8000/api/exploits")


# Singleton instance for easy import across dashboard modules
dashboard_settings = DashboardSettings()
