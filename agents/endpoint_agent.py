# sentenial-x/agents/endpoint_agent.py
import time
from .base_agent import BaseAgent

class EndpointAgent(BaseAgent):
    """
    Concrete endpoint agent implementation.
    """

    def __init__(self, agent_id: str, meta: dict):
        super().__init__(agent_id, meta)
        self.status = "active"
        self.logs_buffer = []

    def heartbeat(self):
        """
        Return agent status for orchestration monitoring.
        """
        return {"agent_id": self.agent_id, "status": self.status, "timestamp": time.time()}

    def report_logs(self, logs: list):
        """
        Append logs to local buffer (could also send to orchestrator via API)
        """
        self.logs_buffer.extend(logs)
        return {"agent_id": self.agent_id, "logs_received": len(logs)}

    def execute_countermeasure(self, action: str):
        """
        Execute a dynamic countermeasure.
        """
        # Example: block IP, isolate process, delete malicious file
        self.logs_buffer.append(f"Countermeasure executed: {action}")
        self.status = f"countermeasure: {action}"
        return {"agent_id": self.agent_id, "action_executed": action}
