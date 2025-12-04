# -*- coding: utf-8 -*-
"""
Sentenial-X Orchestrator Playbook Assembler Module

Responsibilities:
- Assemble playbooks from Jinja2 templates with safe variable substitution.
- Validate a minimal playbook shape before execution.
- Execute playbook steps in a controlled, extensible manner.
- Provide safer conditional evaluation (no raw eval).
- Cleaner logging, error handling and step isolation.

Notes:
- This file keeps external integrations (queue, reflex, ledger, custody, truth_hasher)
  as pluggable collaborators that are awaited where appropriate.
- For security, condition evaluation is intentionally conservative (supports
  simple comparisons, boolean ops, membership, attribute/subscript lookups).
"""

from __future__ import annotations

import ast
import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, List, Callable, Coroutine

from jinja2 import Environment, FileSystemLoader
from cli.memory_adapter import get_adapter
from cli.logger import default_logger
from core.orchestrator.incident_queue import IncidentQueue
from core.orchestrator.incident_reflex_manager import IncidentReflexManager
from core.forensics.ledger_sequencer import LedgerSequencer
from core.forensics.chain_of_custody_builder import ChainOfCustodyBuilder
from core.forensics.truth_vector_hasher import TruthVectorHasher

# Minimal expected playbook keys
REQUIRED_PLAYBOOK_KEYS = {"playbook_id", "name", "triggers", "steps"}

# Allowed AST node types for safe condition evaluation
_ALLOWED_CONDITION_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.BinOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.And,
    ast.Or,
    ast.Not,
    ast.In,
    ast.NotIn,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
)


class PlaybookValidationError(ValueError):
    pass


def _safe_eval_condition(expression: str, context: Dict[str, Any]) -> bool:
    """
    Safely evaluate simple boolean expressions against a provided context.
    Supports comparisons, boolean operators (and/or/not), membership, attribute
    and subscript lookups. No function calls or arbitrary code execution.

    Examples supported:
      - "incident['risk_level'] == 'critical'"
      - "'jailbreak' in incident.get('detections', [])"
      - "incident.get('risk_score', 0) > 8 and 'jailbreak' in incident['detections']"

    Important: This evaluator intentionally disallows CALL nodes to avoid
    executing arbitrary functions. If you need richer expressions, swap to a
    vetted expression language (e.g., jmespath, expr).
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise PlaybookValidationError(f"Invalid condition syntax: {e}")

    # Validate AST nodes
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_CONDITION_NODES):
            raise PlaybookValidationError(f"Disallowed expression element: {node.__class__.__name__}")

        # Explicitly disallow Call nodes (function calls)
        if isinstance(node, ast.Call):
            raise PlaybookValidationError("Function calls are not allowed in playbook conditions")

    # Evaluate with a safe resolver that supports Names, Attributes, and Subscripts
    def _resolve(node):
        if isinstance(node, ast.Expression):
            return _resolve(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            # Name like 'incident' should be taken from context
            if node.id in context:
                return context[node.id]
            raise PlaybookValidationError(f"Unknown name '{node.id}' in condition")
        if isinstance(node, ast.Attribute):
            value = _resolve(node.value)
            return getattr(value, node.attr) if hasattr(value, node.attr) else value.get(node.attr) if isinstance(value, dict) else None
        if isinstance(node, ast.Subscript):
            container = _resolve(node.value)
            idx = _resolve(node.slice) if hasattr(node, "slice") else _resolve(node.slice)
            try:
                return container[idx]
            except Exception:
                return None
        if isinstance(node, ast.Index):  # type: ignore # py38 compat
            return _resolve(node.value)
        if isinstance(node, ast.Tuple):
            return tuple(_resolve(elt) for elt in node.elts)
        if isinstance(node, ast.List):
            return [_resolve(elt) for elt in node.elts]
        if isinstance(node, ast.Dict):
            return {_resolve(k): _resolve(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _resolve(node.operand)
        if isinstance(node, ast.BoolOp):
            values = [_resolve(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise PlaybookValidationError("Unsupported boolean operator in condition")
        if isinstance(node, ast.BinOp):
            left = _resolve(node.left)
            right = _resolve(node.right)
            # Support simple concatenation/addition for strings/numbers
            if isinstance(node.op, (ast.Add,)):
                return left + right
            raise PlaybookValidationError("Unsupported binary operator in condition")
        if isinstance(node, ast.Compare):
            left = _resolve(node.left)
            result = True
            for op, comparator in zip(node.ops, node.comparators):
                right = _resolve(comparator)
                if isinstance(op, ast.Eq):
                    result = result and (left == right)
                elif isinstance(op, ast.NotEq):
                    result = result and (left != right)
                elif isinstance(op, ast.Gt):
                    result = result and (left > right)
                elif isinstance(op, ast.GtE):
                    result = result and (left >= right)
                elif isinstance(op, ast.Lt):
                    result = result and (left < right)
                elif isinstance(op, ast.LtE):
                    result = result and (left <= right)
                elif isinstance(op, ast.In):
                    result = result and (left in right if not isinstance(right, (int, float)) else False)
                elif isinstance(op, ast.NotIn):
                    result = result and (left not in right)
                else:
                    raise PlaybookValidationError(f"Unsupported comparator: {op}")
                left = right
            return result
        raise PlaybookValidationError(f"Unsupported expression node: {type(node).__name__}")

    return bool(_resolve(tree))


class PlaybookAssembler:
    """
    Assemble and execute playbooks built as JSON templates.

    - template_dir: directory for Jinja2 templates (expects JSON output).
    - queue, reflex: optional pluggable collaborators for incident handling.
    """

    def __init__(
        self,
        template_dir: str = "playbooks/templates/",
        queue: Optional[IncidentQueue] = None,
        reflex: Optional[IncidentReflexManager] = None,
    ) -> None:
        # Configure Jinja2 for predictable JSON templating
        self.template_env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)
        self.queue = queue or IncidentQueue()
        self.reflex = reflex or IncidentReflexManager(self.queue)
        self.ledger = LedgerSequencer()
        self.custody = ChainOfCustodyBuilder(self.ledger)
        self.truth_hasher = TruthVectorHasher(self.ledger)
        self.mem = get_adapter()
        self.logger = default_logger
        self.active_playbooks: Dict[str, Dict[str, Any]] = {}

        # Map action name -> handler coroutine
        self._action_handlers: Dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {
            "enqueue": self._action_enqueue,
            "trigger_reflex": self._action_trigger_reflex,
            "log_to_ledger": self._action_log_to_ledger,
            "build_custody": self._action_build_custody,
            "hash_truth": self._action_hash_truth,
            "escalate_incident": self._action_escalate_incident,
            "finalize_custody": self._action_finalize_custody,
            "verify_integrity": self._action_verify_integrity,
            "notify_stakeholder": self._action_notify_stakeholder,
            "quarantine_user": self._action_quarantine_user,
            "analyze_truth_vector": self._action_analyze_truth_vector,
            "add_external_signal": self._action_add_external_signal,
            "check_escalations": self._action_check_escalations,
            "monitor_escalation": self._action_monitor_escalation,
        }

    def _validate_playbook_shape(self, playbook: Dict[str, Any]) -> None:
        if not isinstance(playbook, dict):
            raise PlaybookValidationError("Playbook must be a JSON object")
        missing = REQUIRED_PLAYBOOK_KEYS - set(playbook.keys())
        if missing:
            raise PlaybookValidationError(f"Playbook missing required keys: {missing}")
        if not isinstance(playbook.get("steps"), list):
            raise PlaybookValidationError("Playbook 'steps' must be a list")

    async def assemble_playbook(self, template_name: str, variables: Optional[Dict[str, Any]] = None) -> str:
        """
        Render a Jinja2 template and register the assembled playbook.

        Returns playbook_id.
        """
        try:
            template = self.template_env.get_template(template_name)
        except Exception as e:
            self.logger.error("Template load failed: %s", e)
            raise

        try:
            rendered = template.render(variables or {})
            playbook = json.loads(rendered)
        except json.JSONDecodeError as e:
            self.logger.error("Rendered template is not valid JSON: %s", e)
            raise PlaybookValidationError("Template did not render to valid JSON") from e

        self._validate_playbook_shape(playbook)

        # Ensure stable id
        playbook_id = playbook.get("playbook_id") or f"pb_{uuid.uuid4().hex[:12]}"
        playbook["playbook_id"] = playbook_id
        self.active_playbooks[playbook_id] = playbook

        # Log assembly, best-effort
        try:
            await self.mem.log_command({"action": "assemble_playbook", "playbook_id": playbook_id, "template": template_name})
        except Exception:
            self.logger.debug("Memory adapter log_command failed during assemble; continuing")

        self.logger.info("Assembled playbook %s from %s", playbook_id, template_name)
        return playbook_id

    async def execute_playbook(self, playbook_id: str, incident_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an assembled playbook on the provided incident data.

        Returns a summary with completed steps and any errors encountered.
        """
        if playbook_id not in self.active_playbooks:
            raise KeyError(f"Playbook {playbook_id} not found")

        playbook = self.active_playbooks[playbook_id]
        results: Dict[str, Any] = {"steps_completed": [], "errors": []}
        context: Dict[str, Any] = {"incident": incident_details}

        for idx, step in enumerate(playbook.get("steps", []), start=1):
            action = step.get("action")
            params = step.get("params", {}) or {}
            condition = step.get("condition")

            if not action:
                results["errors"].append(f"Step #{idx} missing 'action'")
                continue

            # Evaluate optional condition safely
            if condition:
                try:
                    cond_ok = _safe_eval_condition(condition, {"incident": context["incident"], **context})
                except PlaybookValidationError as e:
                    results["errors"].append(f"Step #{idx} condition error: {e}")
                    self.logger.error("Playbook %s condition parse error: %s", playbook_id, e)
                    continue
                if not cond_ok:
                    results["steps_completed"].append(f"Skipped {action}: condition false")
                    continue

            handler = self._action_handlers.get(action)
            if handler is None:
                err = f"Unknown action: {action}"
                results["errors"].append(err)
                self.logger.error(err)
                continue

            # Allow per-step timeout (seconds) via params
            timeout = params.get("timeout", None)
            try:
                if timeout:
                    # Guarded call with timeout
                    res = await asyncio.wait_for(handler(playbook_id, context, params), timeout=timeout)
                else:
                    res = await handler(playbook_id, context, params)

                # Handlers may return descriptive strings or dicts
                results["steps_completed"].append(res if isinstance(res, str) else json.dumps(res))
            except asyncio.TimeoutError:
                msg = f"Step {action} timed out after {timeout}s"
                results["errors"].append(msg)
                self.logger.error(msg)
            except Exception as e:
                results["errors"].append(f"Step {action} failed: {e}")
                self.logger.exception("Playbook %s step %s error", playbook_id, action)

        # Final log, best-effort
        try:
            await self.mem.log_command({"action": "execute_playbook", "playbook_id": playbook_id, "results": results})
        except Exception:
            self.logger.debug("Memory adapter log_command failed during execute; continuing")

        self.logger.info("Executed playbook %s: %d completed, %d errors", playbook_id, len(results["steps_completed"]), len(results["errors"]))
        return results

    # ---- Action handlers -------------------------------------------------
    # Each handler accepts (playbook_id, context, params) and returns a descriptive result.

    async def _action_enqueue(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        risk = params.get("risk", "high")
        inc_id = await self.queue.enqueue(risk, context["incident"])
        context["incident_id"] = inc_id
        return f"Enqueued:{inc_id}"

    async def _action_trigger_reflex(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        resp = await self.reflex.trigger_reflex(params.get("risk", "high"), context["incident"])
        context["reflex_results"] = resp
        return f"Reflex:{resp.get('actions_taken', [])}"

    async def _action_log_to_ledger(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        event = await self.ledger.append_event(context["incident"])
        context["ledger_event"] = event
        return f"Logged:{event.get('event_id')}"

    async def _action_build_custody(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        custody_event = await self.custody.build_custody_event(
            actor=params.get("actor", "playbook_executor"),
            action=params.get("custody_action", "executed"),
            evidence_ref=context.get("incident_id", "unknown"),
            description=params.get("desc", "Playbook step."),
            chain_id=params.get("chain_id", f"pb_chain_{playbook_id}")
        )
        context["custody_event"] = custody_event
        return f"Custody:{custody_event.get('custody_id')}"

    async def _action_hash_truth(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        ai_out = context["incident"].get("ai_output")
        if ai_out is None:
            raise ValueError("No 'ai_output' in incident to hash")
        truth_entry = await self.truth_hasher.hash_ai_output(context.get("incident_id", f"pb_out_{uuid.uuid4().hex[:6]}"), ai_out)
        context["truth_vector"] = truth_entry
        return f"Hashed:{truth_entry.get('hash')}"

    async def _action_escalate_incident(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        if "incident_id" not in context:
            raise ValueError("No incident_id in context for escalation")
        success = await self.queue.escalate_incident(context["incident_id"], params.get("new_risk", "critical"))
        return f"Escalated:{success}"

    async def _action_finalize_custody(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        chain_id = params.get("chain_id", f"pb_chain_{playbook_id}")
        summary = await self.custody.finalize_chain(chain_id)
        context["custody_summary"] = summary
        return f"Finalized:{summary.get('chain_id')}"

    async def _action_verify_integrity(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        integrity = await self.ledger.verify_integrity()
        context["integrity_report"] = integrity
        return f"IntegrityVerified:{integrity.get('valid')}"

    async def _action_notify_stakeholder(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        stakeholder = params.get("stakeholder", "security_team")
        # Placeholder for real notification integration
        return f"Notified:{stakeholder}"

    async def _action_quarantine_user(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        user_id = context["incident"].get("user_id", "unknown")
        # Placeholder for real quarantine integration
        return f"Quarantined:{user_id}"

    async def _action_analyze_truth_vector(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        tv = context.get("truth_vector")
        if not tv or "truth_vector" not in tv:
            raise ValueError("No truth_vector in context for analysis")
        vector = tv["truth_vector"]
        avg_fact = float(sum(vector)) / max(1, len(vector))
        context["truth_analysis"] = {"avg_factuality": avg_fact}
        return f"Analyzed:avg_fact={avg_fact:.4f}"

    async def _action_add_external_signal(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        if "incident_id" not in context:
            raise ValueError("No incident_id in context for signal addition")
        signal = params.get("signal", "anomaly_detected")
        success = await getattr(self, "escalator", None).add_external_signal(context["incident_id"], signal) if getattr(self, "escalator", None) else False
        return f"AddedSignal:{signal}:{success}"

    async def _action_check_escalations(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        if getattr(self, "escalator", None):
            await self.escalator._check_escalations()
            return "Manual escalation check triggered"
        return "No escalator configured"

    async def _action_monitor_escalation(self, playbook_id: str, context: Dict[str, Any], params: Dict[str, Any]) -> str:
        # In production, you'd spawn a background task that monitors escalations.
        return "Escalation monitoring enabled (mock)"

    # ---- Utilities ------------------------------------------------------

    async def trigger_on_detection(self, detection: Dict[str, Any]) -> None:
        """
        Trigger playbooks whose triggers match the detection.

        Matching supports:
          - detection['risk_level'] in playbook['triggers']
          - Any detection in detection.get('detections', []) matches a trigger
        """
        if not isinstance(detection, dict):
            self.logger.debug("trigger_on_detection received non-dict: %r", detection)
            return

        for pb_
