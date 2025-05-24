# playbook_assembler.py

from typing import Dict, List
import random

class PlaybookAssembler:
    def __init__(self):
        self.base_playbooks = self._load_base_playbooks()
        self.action_weights = self._load_action_weights()

    def _load_base_playbooks(self) -> Dict[str, List[str]]:
        return {
            "sql_injection": ["log_event", "sanitize_input", "block_ip"],
            "xss": ["log_event", "inject_xss_filter", "notify_admin"],
            "rce": ["log_event", "isolate_host", "terminate_process"],
            "unknown": ["log_event", "increase_monitoring"]
        }

    def _load_action_weights(self) -> Dict[str, int]:
        return {
            "log_event": 1,
            "sanitize_input": 2,
            "inject_xss_filter": 2,
            "block_ip": 3,
            "notify_admin": 4,
            "terminate_process": 5,
            "isolate_host": 5,
            "engage_IR_team": 6,
            "increase_monitoring": 1,
            "defer_to_AI_review": 2,
            "quarantine_asset": 4
        }

    def _score_playbook(self, playbook: List[str]) -> int:
        return sum(self.action_weights.get(action, 0) for action in playbook)

    def assemble(self, threat_type: str, confidence: float, criticality: str, context: Dict = None) -> Dict:
        base = self.base_playbooks.get(threat_type.lower(), self.base_playbooks["unknown"])
        playbook = base.copy()

        if confidence >= 0.9:
            playbook += ["notify_admin"]
            if criticality == "high":
                playbook += ["engage_IR_team", "quarantine_asset"]
        elif 0.6 <= confidence < 0.9:
            playbook += ["increase_monitoring"]
        elif confidence < 0.6:
            playbook = ["log_event", "defer_to_AI_review"]

        if context:
            if context.get("source_reputation") == "malicious":
                playbook.append("block_ip")
            if "cloud" in context.get("attack_vector", ""):
                playbook.append("quarantine_asset")

        score = self._score_playbook(playbook)

        return {
            "threat_type": threat_type,
            "confidence": confidence,
            "criticality": criticality,
            "actions": list(dict.fromkeys(playbook)),  # remove duplicates, preserve order
            "score": score
        }
