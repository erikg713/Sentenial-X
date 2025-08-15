# sentenial-x/agents/config.py

"""
Sentenial-X Agent Configuration
-------------------------------
Centralized configuration for all endpoint agents.
"""

from dataclasses import dataclass
from typing import List, Dict

# ---------------- Heartbeat Settings ----------------
HEARTBEAT_INTERVAL = 5  # seconds between heartbeats to orchestrator
MAX_MISSED_HEARTBEATS = 3

# ---------------- Logging & Telemetry ----------------
LOG_BATCH_SIZE = 10         # number of logs before sending to manager
LOG_FLUSH_INTERVAL = 10     # seconds to flush logs even if batch not full
ENABLE_ENCRYPTION = True    # encrypt logs in transit
ENCRYPTION_KEY = "SENTENIAL_X_SECRET_KEY"  # placeholder, rotate in prod

# ---------------- Countermeasure Defaults ----------------
DEFAULT_COUNTERMEASURES: Dict[str, str] = {
    "malware": "QUARANTINE & ALERT",
    "sql_injection": "BLOCK IP & ALERT",
    "xss": "BLOCK REQUEST & ALERT",
    "normal": "NO_ACTION",
}

# ---------------- Agent Metadata Defaults ----------------
DEFAULT_AGENT_META = {
    "os": "linux",
    "version": "1.0",
    "roles": ["endpoint_monitor"],
    "location": "datacenter_1",
}

# ---------------- Retry / Connectivity ----------------
RETRY_INTERVAL = 3        # seconds between failed API requests
MAX_RETRIES = 5           # max attempts for sending telemetry

# ---------------- Misc ----------------
AGENT_DEBUG = True         # toggle debug logging locally
AGENT_ID_PREFIX = "sx-agent-"  # default ID prefix
