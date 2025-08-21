# cli/alerts.py
import logging
import requests
from datetime import datetime
from .config import AGENT_ID, REPORT_ENDPOINT
from .logger import setup_logger
from .memory import run_async, enqueue_command

logger = setup_logger("alerts")

def dispatch_alert(alert_type: str, severity: str, details: dict):
    """
    Sends an alert to the central server or logs locally.
    """
    payload = {
        "agent_id": AGENT_ID,
        "alert_type": alert_type,
        "severity": severity,
        "details": details,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    try:
        # Example: send to central server (mocked with print if no endpoint)
        if REPORT_ENDPOINT:
            requests.post(REPORT_ENDPOINT, json=payload, timeout=3)
        logger.info(f"Alert dispatched: {alert_type} [{severity}]")
        run_async(enqueue_command(AGENT_ID, "alert_dispatched", payload))
    except Exception as e:
        logger.error(f"Failed to dispatch alert: {e}")
