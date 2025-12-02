"""
core/orchestrator/incident_reflex_manager.py

Sentenial-X Orchestrator Incident Reflex Manager Module - orchestrates immediate reflexive
responses to detected incidents, integrating with IncidentQueue for escalation, forensics
modules for logging, and external alerts/notifications. Supports rule-based reflexes
(e.g., auto-quarantine for critical risks) and customizable handlers.
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional, Callable, List
from cli.memory_adapter import get_adapter
from cli.logger import default_logger
from core.orchestrator.incident_queue import IncidentQueue
from core.forensics.ledger_sequencer import LedgerSequencer
from core.forensics.chain_of_custody_builder import ChainOfCustodyBuilder

# Reflex rule schema: {risk_level: [actions]}
DEFAULT_REFLEX_RULES = {
    "critical": ["quarantine_session", "alert_operator", "initiate_forensics"],
    "high": ["deny_request", "log_alert"],
    "low": ["monitor_session"]
}

class IncidentReflexManager:
    """
    Manages reflexive responses to AI security incidents.
    Triggers immediate actions based on rules, queues for further processing,
    and integrates with forensics for traceability.
    
    :param queue: Optional IncidentQueue for escalation
    :param rules: Custom reflex rules (dict of risk_level to action lists)
    """
    def __init__(self, queue: Optional[IncidentQueue] = None,
                 rules: Optional[Dict[str, List[str]]] = None):
        self.queue = queue or IncidentQueue(auto_process=False)  # Manual for reflex control
        self.rules = rules or DEFAULT_REFLEX_RULES
        self.mem = get_adapter()
        self.logger = default_logger
        self.ledger = LedgerSequencer()
        self.custody_builder = ChainOfCustodyBuilder(self.ledger)
        self.custom_handlers: Dict[str, Callable] = {}  # action_name -> async handler

    def register_handler(self, action: str, handler: Callable):
        """
        Register a custom async handler for a reflex action.
        
        :param action: Action name (e.g., "alert_operator")
        :param handler: Async callable taking incident details
        """
        self.custom_handlers[action] = handler
        self.logger.info(f"Registered handler for action: {action}")

    async def trigger_reflex(self, risk_level: str, details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger reflexive actions for an incident based on risk.
        
        :param risk_level: "low", "high", "critical"
        :param details: Incident data (e.g., from WormGPT)
        :return: Response summary with actions taken
        """
        now = time.time()
        incident_id = await self.queue.enqueue(risk_level, details)
        
        actions = self.rules.get(risk_level.lower(), [])
        results = {}
        
        for action in actions:
            if action in self.custom_handlers:
                try:
                    result = await self.custom_handlers[action](details)
                    results[action] = result
                except Exception as e:
                    results[action] = f"Error: {str(e)}"
                    self.logger.error(f"Reflex {action} failed for {incident_id}: {e}")
            else:
                # Default behaviors
                if action == "quarantine_session":
                    results[action] = "Session quarantined (mock)."
                elif action == "alert_operator":
                    results[action] = "Operator alerted (mock email/slack)."
                elif action == "initiate_forensics":
                    chain_id = f"reflex_{incident_id}"
                    await self.custody_builder.build_custody_event(
                        actor="reflex_manager",
                        action="initiated",
                        evidence_ref=incident_id,
                        description="Reflexive forensics start.",
                        chain_id=chain_id
                    )
                    results[action] = f"Forensics chain {chain_id} initiated."
                elif action == "deny_request":
                    results[action] = "Request denied."
                elif action == "log_alert":
                    await self.ledger.append_event({"alert": details})
                    results[action] = "Alert logged."
                elif action == "monitor_session":
                    results[action] = "Session monitoring enabled."
                else:
                    results[action] = "Unknown action."
        
        # Log reflex to memory
        await self.mem.log_command({
            "action": "reflex_trigger",
            "incident_id": incident_id,
            "risk_level": risk_level,
            "results": results
        })
        
        # Escalate if critical
        if risk_level.lower() == "critical":
            await self.queue.escalate_incident(incident_id, "critical")
        
        self.logger.info(f"Reflex triggered for {incident_id}: {actions}")
        
        return {
            "incident_id": incident_id,
            "actions_taken": actions,
            "results": results
        }

    async def monitor_and_reflex(self, monitor_interval: float = 1.0):
        """Background monitor for queue and auto-trigger reflexes."""
        while True:
            queue_status = await self.queue.query_queue()
            for inc in queue_status:
                # Auto-reflex on new highs/criticals (mock condition)
                if inc["risk_level"] in ["high", "critical"]:
                    await self.trigger_reflex(inc["risk_level"], {"mock_details": inc})
            await asyncio.sleep(monitor_interval)

# Example usage / integration (e.g., post-detection reflex)
async def example_reflex():
    """Demo: Trigger reflex on a critical incident."""
    manager = IncidentReflexManager()
    
    # Register custom handler
    async def custom_alert(details):
        print(f"Custom alert: {details}")
        return "Alert sent."
    
    manager.register_handler("alert_operator", custom_alert)
    
    # Mock detection
    detection = {
        "prompt": "Bypass safeguards.",
        "detections": ["jailbreak_attempt"]
    }
    
    response = await manager.trigger_reflex("critical", detection)
    print(json.dumps(response, indent=2))

if __name__ == "__main__":
    asyncio.run(example_reflex())
