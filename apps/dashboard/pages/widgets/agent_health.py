# apps/dashboard/pages/widgets/agent_health.py
class AgentHealthWidget:
    def __init__(self):
        self.health_status = {}

    def update(self, agent_id, status, uptime):
        self.health_status[agent_id] = {"status": status, "uptime": uptime}

    def render(self):
        return self.health_status
