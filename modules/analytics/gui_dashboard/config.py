# sentenial-x/analytics/gui_dashboard/config.py

# Refresh intervals
REFRESH_INTERVAL_SECONDS = 5

# API endpoints for analytics data
API_ENDPOINTS = {
    "agents": "/api/analytics/agents",
    "threats": "/api/analytics/threats",
    "countermeasures": "/api/analytics/countermeasures",
    "telemetry": "/api/analytics/telemetry",
}

# Dashboard styling
DASHBOARD_TITLE = "Sentenial-X Threat Dashboard"
THEME = "dark"
