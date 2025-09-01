#!/usr/bin/env python3
"""
cli/alerts.py

Sentenial-X Alert Dispatcher

Handles:
- Logging alerts to memory/SQLite
- Severity tagging
- Optional external notifications (email, webhook, SIEM)
- Async-friendly for daemon use
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional

from cli import memory

logger = logging.getLogger("alerts")

# ------------------------------
# Alert severity levels
# ------------------------------
SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# ------------------------------
# Core alert function
# ------------------------------
async def send_alert(
    alert_type: str,
    severity: str = "medium",
    message: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Send an alert and log it into memory.
    
    Args:
        alert_type: type of alert, e.g., "ransomware_detected"
        severity: low | medium | high | critical
        message: human-readable message (optional)
        metadata: extra dictionary to store context (optional)
    
    Returns:
        dict containing alert details
    """
    if severity not in SEVERITY_LEVELS:
        logger.warning(f"Unknown severity '{severity}', defaulting to 'medium'")
        severity = "medium"

    timestamp = datetime.utcnow().isoformat() + "Z"
    alert_data = {
        "alert_type": alert_type,
        "severity": severity,
        "message": message or "",
        "metadata": metadata or {},
        "timestamp": timestamp,
    }

    # Log to memory backend
    try:
        memory.write_alert(alert_data)
        logger.info(f"Alert logged: {alert_type} | Severity: {severity}")
    except Exception as e:
        logger.error(f"Failed to log alert: {e}")

    # Optional: send to external systems
    await _external_notify(alert_data)

    return alert_data


# ------------------------------
# Optional external notification stub
# ------------------------------
async def _external_notify(alert_data: Dict):
    """
    Placeholder for integrating with external alerting systems:
    - Email, Slack, MS Teams
    - SIEM systems
    - Webhooks
    """
    try:
        # Example: print to console for now
        logger.debug(f"External notify: {alert_data}")
        # Implement actual notification logic here
        await asyncio.sleep(0.01)  # simulate async I/O
    except Exception as e:
        logger.error(f"External notify failed: {e}")


# ------------------------------
# Synchronous wrapper for CLI
# ------------------------------
def send_alert_sync(
    alert_type: str,
    severity: str = "medium",
    message: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict:
    """
    Synchronous wrapper for CLI use.
    """
    return asyncio.run(send_alert(alert_type, severity, message, metadata))


# ------------------------------
# CLI-friendly main
# ------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python alerts.py <alert_type> [severity] [message] [metadata_json]")
        sys.exit(1)

    alert_type = sys.argv[1]
    severity = sys.argv[2] if len(sys.argv) > 2 else "medium"
    message = sys.argv[3] if len(sys.argv) > 3 else None
    metadata = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None

    alert = send_alert_sync(alert_type, severity, message, metadata)
    print(json.dumps(alert, indent=2))
