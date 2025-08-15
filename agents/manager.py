# sentenial-x/agents/manager.py
from typing import Dict
from .endpoint_agent import EndpointAgent

class AgentManager:
    """
    Central manager for all endpoint agents.
    """

    def __init__(self):
        self.agents: Dict[str, EndpointAgent] = {}

    def register(self, agent_id: str, meta: dict):
        if agent_id not in self.agents:
            self.agents[agent_id] = EndpointAgent(agent_id, meta)
        return self.agents[agent_id]

    def get_agent(self, agent_id: str):
        return self.agents.get(agent_id)

    def heartbeat_all(self):
        return {aid: agent.heartbeat() for aid, agent in self.agents.items()}

    def report_all_logs(self):
        all_logs = {}
        for aid, agent in self.agents.items():
            all_logs[aid] = agent.logs_buffer.copy()
            agent.logs_buffer.clear()
        return all_logs

    def deploy_countermeasure(self, agent_id: str, action: str):
        agent = self.agents.get(agent_id)
        if agent:
            return agent.execute_countermeasure(action)
        return None
