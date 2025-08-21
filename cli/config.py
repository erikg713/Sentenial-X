# cli/config.py
import os
from pathlib import Path

# -----------------------------
# General Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_DIR = BASE_DIR / "data" / "logs"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
REPORTS_DIR = BASE_DIR / "data" / "reports"
AGENT_MEMORY_DB = BASE_DIR / "data" / "memory.db"

# -----------------------------
# Agent Configuration
# -----------------------------
AGENT_ID = os.getenv("SENTENIAL_AGENT_ID", "agent-001")
AGENT_NAME = os.getenv("SENTENIAL_AGENT_NAME", "SentenialAgent")
AGENT_MODE = os.getenv("SENTENIAL_AGENT_MODE", "passive")  # passive / active / autonomous

# -----------------------------
# CLI Defaults
# -----------------------------
DEFAULT_WORMGPT_TEMPERATURE = float(os.getenv("WORMGPT_TEMPERATURE", 0.7))
DEFAULT_TELEMETRY_FILTER = os.getenv("TELEMETRY_FILTER", None)
DEFAULT_ALERT_SEVERITY = os.getenv("DEFAULT_ALERT_SEVERITY", "medium")

# -----------------------------
# Telemetry / Reporting
# -----------------------------
TELEMETRY_ENABLED = os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"
TELEMETRY_ENDPOINT = os.getenv("TELEMETRY_ENDPOINT", "http://localhost:5000/telemetry")
REPORT_ENDPOINT = os.getenv("REPORT_ENDPOINT", "http://localhost:5000/report")

# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = LOGS_DIR / "cli_agent.log"

# -----------------------------
# ML / Threat Modules
# -----------------------------
ML_MODELS_DIR = BASE_DIR / "models"
BERT_MODEL_PATH = ML_MODELS_DIR / "bert_intent_classifier"
LORE_MODEL_DIR = ML_MODELS_DIR / "lora"
ENCODER_MODEL_DIR = ML_MODELS_DIR / "encoder"

# -----------------------------
# Helper Functions
# -----------------------------
def ensure_dirs():
    """Ensure all essential directories exist."""
    for path in [LOGS_DIR, EMBEDDINGS_DIR, REPORTS_DIR, ML_MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
