# sentenial-x/agents/retaliation_bot/logger.py
import time
from typing import Optional, Dict
from ..telemetry import TelemetryBuffer
from ..config import ENABLE_ENCRYPTION, ENCRYPTION_KEY

class RetaliationLogger:
    """
    Structured logging for RetaliationBot.
    Sends telemetry securely to orchestrator.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.telemetry = TelemetryBuffer(agent_id)

    def log_action(self, action: str, log_source: Optional[str] = None, meta: Optional[Dict] = None):
        """
        Log a countermeasure action executed by the bot.
        """
        entry_meta = meta or {}
        if log_source:
            entry_meta["log_source"] = log_source

        entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Action: {action}"
        self.telemetry.add_log(entry, meta=entry_meta)

    def log_info(self, message: str, meta: Optional[Dict] = None):
        """
        General info logging.
        """
        entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] INFO: {message}"
        self.telemetry.add_log(entry, meta=meta or {})

    def log_warning(self, message: str, meta: Optional[Dict] = None):
        """
        Warning logs (potential anomalies detected).
        """
        entry = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] WARNING: {message}"
        self.telemetry.add_log(entry, meta=meta or {})

    def flush(self):
        """
        Force flush of buffered logs to orchestrator.
        """
        self.telemetry.flush()
