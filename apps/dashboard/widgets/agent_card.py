# apps/dashboard/widgets/agent_card.py
class AgentCardWidget:
    def __init__(self):
        self.agents = {}

    def update(self, agent_id, status, timestamp):
        self.agents[agent_id] = {"status": status, "timestamp": timestamp}

    def render(self):
        return self.agents
