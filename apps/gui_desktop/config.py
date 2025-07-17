import os
from pathlib import Path

# Root directory for temporary test sandboxes
SANDBOX_ROOT = Path(os.getenv("SENTENIAL_SANDBOX_DIR", Path.home() / "SentenialSandbox"))

# Default payload to load on startup
DEFAULT_PAYLOAD = "basic_encrypt"

# Default number of test files
DEFAULT_FILE_COUNT = 10

# Enable developer mode to show extra debugging tools in GUI
DEVELOPER_MODE = os.getenv("SENTENIAL_DEV_MODE", "false").lower() == "true"

# GUI Theme
GUI_THEME = "dark"  # Options: 'dark', 'light'

# Monitoring enabled by default
DEFAULT_MONITORING_ENABLED = True

# UI Labels and Hints
UI_LABELS = {
    "window_title": "Sentenial X – Ransomware Emulator",
    "payload_dropdown": "Select Payload:",
    "file_count": "Number of Test Files:",
    "monitor_checkbox": "Enable Monitoring",
    "run_button": "Run Emulation",
    "results_label": "Results:",
    "output_placeholder": "Output will appear here...",
}

# Version
APP_VERSION = "1.0.0"
