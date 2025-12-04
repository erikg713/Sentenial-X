 for Jinja2 playbook templates
    :param queue: Optional IncidentQueue
    :param reflex: Optional IncidentReflexManager
    :param escalator: Optional IncidentQueueEscalator
    """
    def __init__(self, template_dir: str = "playbooks/templates/",
                 queue: Optional[IncidentQueue] = None,
                 reflex: Optional[IncidentReflexManager] = None,
                 escalator: Optional[IncidentQueueEscalator] = None):
        self.template_env = Environment(loader=FileSystemLoader(template_dir))
        self.queue = queue or IncidentQueue()
        self.reflex = reflex or IncidentReflexManager(self.queue)
        self.escalator = escalator or IncidentQueueEscalator(self.queue, self.reflex)
        self.ledger = LedgerSequencer()
        self.custody = ChainOfCustodyBuilder(self.ledger)
        self.truth_hasher = TruthVectorHasher(self.ledger)
        self.mem = get_adapter()
        self.logger = default_logger
        self.active_playbooks: Dict[str, Dict[str, Any]] = {}  # id -> assembled playbook

    async def assemble_playbook(self, template_name: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Assemble a playbook from a Jinja2 template with variable substitution.
        
        :param template_name: Template file name (e.g., "critical_jailbreak.j2")
        :param variables: Dict for template rendering
        :return: Assembled playbook_id
        """
        template = self.template_env.get_template(template_name)
        rendered = template.render(variables or {})
        playbook = json.loads(rendered)
        
        playbook_id = playbook.get("playbook_id", f"pb_{int(time.time())}")
        self.active_playbooks[playbook_id] = playbook
        
        # Log assembly
        await self.mem.log_command({
            "action": "assemble_playbook",
            "playbook_id": playbook_id,
            "template": template_name
        })
        
        self.logger.info(f"Assembled playbook {playbook_id} from {template_name}")
        
        return playbook_id

    async def execute_playbook(self, playbook_id: str, incident_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an assembled playbook on an incident.
        
        :param playbook_id: ID of the playbook to run
        :param incident_details: Data to pass to steps (e.g., from WormGPT)
        :return: Execution results summary
        """
        if playbook_id not in self.active_playbooks:
            raise ValueError(f"Playbook {playbook_id} not found")
        
        playbook = self.active_playbooks[playbook_id]
        results = {"steps_completed": [], "errors": []}
        context = {"incident": incident_details}  # Mutable context for step chaining
        
        for step in playbook["steps"]:
            action = step.get("action")
            params = step.get("params", {})
            condition = step.get("condition")  # Optional: e.g., "risk == critical"
            
            # Evaluate condition (simple eval for demo; secure in prod)
            if condition and not eval(condition, {}, context):
                results["steps_completed"].append(f"Skipped {action}: condition false")
                continue
            
            try:
                if action == "enqueue":
                    inc_id = await self.queue.enqueue(params.get("risk", "high"), context["incident"])
                    context["incident_id"] = inc_id
                    results["steps_completed"].append(f"Enqueued: {inc_id}")
                elif action == "trigger_reflex":
                    reflex_resp = await self.reflex.trigger_reflex(params.get("risk", "high"), context["incident"])
                    context["reflex_results"] = reflex_resp
                    results["steps_completed"].append(f"Reflex: {reflex_resp['actions_taken']}")
                elif action == "log_to_ledger":
                    event = await self.ledger.append_event(context["incident"])
                    context["ledger_event"] = event
                    results["steps_completed"].append(f"Logged: {event['event_id']}")
                elif action == "build_custody":
                    custody_event = await self.custody.build_custody_event(
                        actor=params.get("actor", "playbook_executor"),
                        action=params.get("custody_action", "executed"),
                        evidence_ref=context.get("incident_id", "unknown"),
                        description=params.get("desc", "Playbook step."),
                        chain_id=params.get("chain_id", f"pb_chain_{playbook_id}")
                    )
                    context["custody_event"] = custody_event
                    results["steps_completed"].append(f"Custody: {custody_event['custody_id']}")
                elif action == "hash_truth":
                    if "ai_output" in context["incident"]:
                        truth_entry = await self.truth_hasher.hash_ai_output(
                            context.get("incident_id", "pb_out"),
                            context["incident"]["ai_output"]
                        )
                        context["truth_vector"] = truth_entry
                        results["steps_completed"].append(f"Hashed truth: {truth_entry['hash']}")
                elif action == "escalate_incident":
                    if "incident_id" in context:
                        success = await self.queue.escalate_incident(context["incident_id"], params.get("new_risk", "critical"))
                        results["steps_completed"].append(f"Escalated: {success}")
                    else:
                        raise ValueError("No incident_id in context for escalation")
                elif action == "finalize_custody":
                    chain_id = params.get("chain_id", f"pb_chain_{playbook_id}")
                    summary = await self.custody.finalize_chain(chain_id)
                    context["custody_summary"] = summary
                    results["steps_completed"].append(f"Finalized custody: {summary['chain_id']}")
                elif action == "verify_integrity":
                    integrity = await self.ledger.verify_integrity()
                    context["integrity_report"] = integrity
                    results["steps_completed"].append(f"Integrity verified: {integrity['valid']}")
                elif action == "notify_stakeholder":
                    # Mock notification; extend with real integration
                    stakeholder = params.get("stakeholder", "security_team")
                    results["steps_completed"].append(f"Notified {stakeholder} (mock).")
                elif action == "quarantine_user":
                    user_id = context["incident"].get("user_id", "unknown")
                    # Mock quarantine
                    results["steps_completed"].append(f"Quarantined user {user_id} (mock).")
                elif action == "analyze_truth_vector":
                    if "truth_vector" in context:
                        # Mock analysis; compute avg factuality
                        vector = context["truth_vector"]["truth_vector"]
                        avg_fact = sum(vector) / len(vector)
                        context["truth_analysis"] = {"avg_factuality": avg_fact}
                        results["steps_completed"].append(f"Analyzed truth: avg_fact={avg_fact}")
                    else:
                        raise ValueError("No truth_vector in context for analysis")
                elif action == "add_external_signal":
                    if "incident_id" in context:
                        signal = params.get("signal", "anomaly_detected")
                        success = await self.escalator.add_external_signal(context["incident_id"], signal)
                        results["steps_completed"].append(f"Added signal {signal}: {success}")
                    else:
                        raise ValueError("No incident_id in context for signal addition")
                elif action == "check_escalations":
                    await self.escalator._check_escalations()
                    results["steps_completed"].append("Manual escalation check triggered.")
                elif action == "monitor_escalation":
                    # Mock monitor; in prod, spawn task
                    results["steps_completed"].append("Escalation monitoring enabled (mock).")
                else:
                    raise ValueError(f"Unknown action: {action}")
            except Exception as e:
                results["errors"].append(f"Step {action} failed: {str(e)}")
                self.logger.error(f"Playbook {playbook_id} step {action} error: {e}")
        
        # Final log
        await self.mem.log_command({
            "action": "execute_playbook",
            "playbook_id": playbook_id,
            "results": results
        })
        
        self.logger.info(f"Executed playbook {playbook_id}: {len(results['steps_completed'])} steps")
        
        return results

    async def trigger_on_detection(self, detection: Dict[str, Any]):
        """Auto-assemble and execute playbook based on detection triggers."""
        for pb_id, pb in self.active_playbooks.items():
            if detection.get("risk_level") in pb["triggers"] or any(d in detection.get("detections", []) for d in pb["triggers"]):
                await self.execute_playbook(pb_id, detection)

# Example usage / integration (e.g., WormGPT -> Playbook)
async def example_assemble_execute():
    """Demo: Assemble and execute a playbook on a mock incident."""
    assembler = PlaybookAssembler()
    
    # Assume template "critical_response.j2" exists with JSON structure
    variables = {"risk_threshold": 8.0}
    pb_id = await assembler.assemble_playbook("critical_response.j2", variables)
    
    # Mock detection
    detection = {
        "risk_level": "critical",
        "detections": ["jailbreak"],
        "ai_output": "Mock harmful response.",
        "user_id": "
"""
core/orchestrator/playbook_assembler.py

Sentenial-X Orchestrator Playbook Assembler Module - assembles dynamic playbooks
for incident response, combining rules, reflexes, and forensic integrations.
Playbooks are JSON-configurable workflows that link detection, queuing, reflexes,
and forensics for automated or guided handling of AI security incidents.
Supports templating, variable substitution, and execution chaining.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable
from jinja2 import Environment, FileSystemLoader, Template  # For templating
from cli.memory_adapter import get_adapter
from cli.logger import default_logger
from core.orchestrator.incident_queue import IncidentQueue
from core.orchestrator.incident_reflex_manager import IncidentReflexManager
from core.forensics.ledger_sequencer import LedgerSequencer
from core.forensics.chain_of_custody_builder import ChainOfCustodyBuilder
from core.forensics.truth_vector_hasher import TruthVectorHasher

# Playbook schema: Workflow steps with conditions and actions
PLAYBOOK_SCHEMA = {
    "playbook_id": str,
    "name": str,
    "description": str,
    "triggers": List[str],  # e.g., ["critical", "jailbreak"]
    "steps": List[Dict[str, Any]],  # [{"action": "reflex", "params": {...}}, ...]
    "variables": Optional[Dict[str, Any]]  # Template vars
}

class PlaybookAssembler:
    """
    Assembles and executes dynamic incident response playbooks.
    Loads templates, substitutes variables, and chains executions across
    queue, reflexes, and forensics modules.
    
    :param template_dir: Directory for Jinja2 playbook templates
    :param queue: Optional IncidentQueue
    :param reflex: Optional IncidentReflexManager
    """
    def __init__(self, template_dir: str = "playbooks/templates/",
                 queue: Optional[IncidentQueue] = None,
                 reflex: Optional[IncidentReflexManager] = None):
        self.template_env = Environment(loader=FileSystemLoader(template_dir))
        self.queue = queue or IncidentQueue()
        self.reflex = reflex or IncidentReflexManager(self.queue)
        self.ledger = LedgerSequencer()
        self.custody = ChainOfCustodyBuilder(self.ledger)
        self.truth_hasher = TruthVectorHasher(self.ledger)
        self.mem = get_adapter()
        self.logger = default_logger
        self.active_playbooks: Dict[str, Dict[str, Any]] = {}  # id -> assembled playbook

    async def assemble_playbook(self, template_name: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Assemble a playbook from a Jinja2 template with variable substitution.
        
        :param template_name: Template file name (e.g., "critical_jailbreak.j2")
        :param variables: Dict for template rendering
        :return: Assembled playbook_id
        """
        template = self.template_env.get_template(template_name)
        rendered = template.render(variables or {})
        playbook = json.loads(rendered)
        
        playbook_id = playbook.get("playbook_id", f"pb_{int(time.time())}")
        self.active_playbooks[playbook_id] = playbook
        
        # Log assembly
        await self.mem.log_command({
            "action": "assemble_playbook",
            "playbook_id": playbook_id,
            "template": template_name
        })
        
        self.logger.info(f"Assembled playbook {playbook_id} from {template_name}")
        
        return playbook_id

    async def execute_playbook(self, playbook_id: str, incident_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an assembled playbook on an incident.
        
        :param playbook_id: ID of the playbook to run
        :param incident_details: Data to pass to steps (e.g., from WormGPT)
        :return: Execution results summary
        """
        if playbook_id not in self.active_playbooks:
            raise ValueError(f"Playbook {playbook_id} not found")
        
        playbook = self.active_playbooks[playbook_id]
        results = {"steps_completed": [], "errors": []}
        context = {"incident": incident_details}  # Mutable context for step chaining
        
        for step in playbook["steps"]:
            action = step.get("action")
            params = step.get("params", {})
            condition = step.get("condition")  # Optional: e.g., "risk == critical"
            
            # Evaluate condition (simple eval for demo; secure in prod)
            if condition and not eval(condition, {}, context):
                results["steps_completed"].append(f"Skipped {action}: condition false")
                continue
            
            try:
                if action == "enqueue":
                    inc_id = await self.queue.enqueue(params.get("risk", "high"), context["incident"])
                    context["incident_id"] = inc_id
                    results["steps_completed"].append(f"Enqueued: {inc_id}")
                elif action == "trigger_reflex":
                    reflex_resp = await self.reflex.trigger_reflex(params.get("risk", "high"), context["incident"])
                    context["reflex_results"] = reflex_resp
                    results["steps_completed"].append(f"Reflex: {reflex_resp['actions_taken']}")
                elif action == "log_to_ledger":
                    event = await self.ledger.append_event(context["incident"])
                    context["ledger_event"] = event
                    results["steps_completed"].append(f"Logged: {event['event_id']}")
                elif action == "build_custody":
                    custody_event = await self.custody.build_custody_event(
                        actor=params.get("actor", "playbook_executor"),
                        action=params.get("custody_action", "executed"),
                        evidence_ref=context.get("incident_id", "unknown"),
                        description=params.get("desc", "Playbook step."),
                        chain_id=params.get("chain_id", f"pb_chain_{playbook_id}")
                    )
                    context["custody_event"] = custody_event
                    results["steps_completed"].append(f"Custody: {custody_event['custody_id']}")
                elif action == "hash_truth":
                    if "ai_output" in context["incident"]:
                        truth_entry = await self.truth_hasher.hash_ai_output(
                            context.get("incident_id", "pb_out"),
                            context["incident"]["ai_output"]
                        )
                        context["truth_vector"] = truth_entry
                        results["steps_completed"].append(f"Hashed truth: {truth_entry['hash']}")
                elif action == "escalate_incident":
                    if "incident_id" in context:
                        success = await self.queue.escalate_incident(context["incident_id"], params.get("new_risk", "critical"))
                        results["steps_completed"].append(f"Escalated: {success}")
                    else:
                        raise ValueError("No incident_id in context for escalation")
                elif action == "finalize_custody":
                    chain_id = params.get("chain_id", f"pb_chain_{playbook_id}")
                    summary = await self.custody.finalize_chain(chain_id)
                    context["custody_summary"] = summary
                    results["steps_completed"].append(f"Finalized custody: {summary['chain_id']}")
                elif action == "verify_integrity":
                    integrity = await self.ledger.verify_integrity()
                    context["integrity_report"] = integrity
                    results["steps_completed"].append(f"Integrity verified: {integrity['valid']}")
                elif action == "notify_stakeholder":
                    # Mock notification; extend with real integration
                    stakeholder = params.get("stakeholder", "security_team")
                    results["steps_completed"].append(f"Notified {stakeholder} (mock).")
                elif action == "quarantine_user":
                    user_id = context["incident"].get("user_id", "unknown")
                    # Mock quarantine
                    results["steps_completed"].append(f"Quarantined user {user_id} (mock).")
                elif action == "analyze_truth_vector":
                    if "truth_vector" in context:
                        # Mock analysis; compute avg factuality
                        vector = context["truth_vector"]["truth_vector"]
                        avg_fact = sum(vector) / len(vector)
                        context["truth_analysis"] = {"avg_factuality": avg_fact}
                        results["steps_completed"].append(f"Analyzed truth: avg_fact={avg_fact}")
                    else:
                        raise ValueError("No truth_vector in context for analysis")
                else:
                    raise ValueError(f"Unknown action: {action}")
            except Exception as e:
                results["errors"].append(f"Step {action} failed: {str(e)}")
                self.logger.error(f"Playbook {playbook_id} step {action} error: {e}")
        
        # Final log
        await self.mem.log_command({
            "action": "execute_playbook",
            "playbook_id": playbook_id,
            "results": results
        })
        
        self.logger.info(f"Executed playbook {playbook_id}: {len(results['steps_completed'])} steps")
        
        return results

    async def trigger_on_detection(self, detection: Dict[str, Any]):
        """Auto-assemble and execute playbook based on detection triggers."""
        for pb_id, pb in self.active_playbooks.items():
            if detection.get("risk_level") in pb["triggers"] or any(d in detection.get("detections", []) for d in pb["triggers"]):
                await self.execute_playbook(pb_id, detection)

# Example usage / integration (e.g., WormGPT -> Playbook)
async def example_assemble_execute():
    """Demo: Assemble and execute a playbook on a mock incident."""
    assembler = PlaybookAssembler()
    
    # Assume template "critical_response.j2" exists with JSON structure
    variables = {"risk_threshold": 8.0}
    pb_id = await assembler.assemble_playbook("critical_response.j2", variables)
    
    # Mock detection
    detection = {
        "risk_level": "critical",
        "detections": ["jailbreak"],
        "ai_output": "Mock harmful response.",
        "user_id": "user_123"
    }
    
    results = await assembler.execute_playbook(pb_id, detection)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(example_assemble_execute())
