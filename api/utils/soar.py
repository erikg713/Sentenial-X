# api/utils/soar.py

"""
SOAR (Security Orchestration, Automation, and Response) utilities.

This module provides integrations with external SOAR platforms,
ticketing systems, and internal automation playbooks. It allows
for automated incident response, enrichment, and escalation.
"""

import json
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional, List

from api.utils.logger import get_logger

logger = get_logger(__name__)


class SOARClient:
    """
    A client for interacting with external SOAR platforms and
    automating incident response workflows.
    """

    def __init__(self, base_url: str, api_key: str, verify_ssl: bool = True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Internal request handler for SOAR API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(
                method=method.upper(),
                url=url,
                headers=self.headers,
                json=data,
                verify=self.verify_ssl,
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"SOAR API request failed: {e}")
            return {"error": str(e)}

    def create_incident(self, title: str, description: str, severity: str, artifacts: List[Dict]) -> Dict:
        """Create an incident in the SOAR system."""
        payload = {
            "title": title,
            "description": description,
            "severity": severity,
            "artifacts": artifacts,
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.info(f"Creating SOAR incident: {title} (severity={severity})")
        return self._request("POST", "/incidents", data=payload)

    def update_incident(self, incident_id: str, updates: Dict[str, Any]) -> Dict:
        """Update an existing incident in the SOAR system."""
        logger.info(f"Updating SOAR incident {incident_id} with {updates}")
        return self._request("PATCH", f"/incidents/{incident_id}", data=updates)

    def close_incident(self, incident_id: str, resolution: str) -> Dict:
        """Close an incident with resolution notes."""
        payload = {"status": "closed", "resolution": resolution}
        logger.info(f"Closing SOAR incident {incident_id} with resolution: {resolution}")
        return self._request("PATCH", f"/incidents/{incident_id}", data=payload)

    def run_playbook(self, playbook_name: str, context: Dict) -> Dict:
        """Execute a SOAR playbook with given context."""
        payload = {"playbook": playbook_name, "context": context}
        logger.info(f"Executing playbook {playbook_name} with context: {context}")
        return self._request("POST", "/playbooks/run", data=payload)


class PlaybookEngine:
    """
    Internal playbook engine for automated responses without
    requiring an external SOAR.
    """

    def __init__(self):
        self.playbooks = {}

    def register_playbook(self, name: str, func):
        """Register a playbook function by name."""
        logger.debug(f"Registering playbook: {name}")
        self.playbooks[name] = func

    def run(self, name: str, context: Dict) -> Dict:
        """Run a registered playbook with context."""
        if name not in self.playbooks:
            logger.warning(f"Playbook {name} not found")
            return {"status": "error", "message": f"Playbook {name} not registered"}
        try:
            logger.info(f"Running playbook {name} with context: {context}")
            return self.playbooks[name](context)
        except Exception as e:
            logger.error(f"Error running playbook {name}: {e}")
            return {"status": "error", "message": str(e)}


# Example built-in playbooks
def isolate_host_playbook(context: Dict) -> Dict:
    """
    Example playbook: Isolate a compromised host.
    In real deployments, this could trigger EDR integrations.
    """
    host = context.get("host")
    logger.info(f"Simulated: Isolating host {host}")
    return {"status": "success", "action": f"Host {host} isolated"}


def disable_account_playbook(context: Dict) -> Dict:
    """
    Example playbook: Disable a compromised user account.
    """
    user = context.get("user")
    logger.info(f"Simulated: Disabling account {user}")
    return {"status": "success", "action": f"Account {user} disabled"}


# Initialize default engine with sample playbooks
playbook_engine = PlaybookEngine()
playbook_engine.register_playbook("isolate_host", isolate_host_playbook)
playbook_engine.register_playbook("disable_account", disable_account_playbook)
