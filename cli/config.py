import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# ---------------------------
# Agent Configuration
# ---------------------------
AGENT_ID = os.getenv("AGENT_ID", "sentenial_agent_001")
AGENT_NAME = os.getenv("AGENT_NAME", "AI_Operator")
AGENT_ROLE = os.getenv("AGENT_ROLE", "full_operator")

# ---------------------------
# Memory / Database Configuration
# ---------------------------
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "sqlite")  # options: sqlite, postgres, redis
DB_PATH = os.getenv("DB_PATH", "sentenial.db")
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/sentenial_db")

# ---------------------------
# Logging Configuration
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = os.getenv("LOG_FILE", "logs/sentenial.log")  # ensure logs/ folder exists

# ---------------------------
# Telemetry Configuration
# ---------------------------
TELEMETRY_ENABLED = os.getenv("TELEMETRY_ENABLED", "True").lower() in ("true", "1", "yes")
TELEMETRY_SOURCES = os.getenv("TELEMETRY_SOURCES", "network_monitor,endpoint_sensor").split(",")

# ---------------------------
# Orchestrator Configuration
# ---------------------------
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
ORCHESTRATOR_TOKEN = os.getenv("ORCHESTRATOR_TOKEN", "changeme12345")

# ---------------------------
# WormGPT / Detection
# ---------------------------
WORMGPT_TEMPERATURE = float(os.getenv("WORMGPT_TEMPERATURE", 0.7))

# ---------------------------
# Alerts Configuration
# ---------------------------
ALERT_DEFAULT_SEVERITY = os.getenv("ALERT_DEFAULT_SEVERITY", "medium")

# ---------------------------
# Simulation / Testing
# ---------------------------
SIMULATOR_ENABLED = os.getenv("SIMULATOR_ENABLED", "True").lower() in ("true", "1", "yes")

# ---------------------------
# External API Keys (Optional)
# ---------------------------
# EXTERNAL_API_KEY = os.getenv("EXTERNAL_API_KEY", None)
