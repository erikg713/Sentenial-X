# apps/dashboard/widgets/countermeasure_log.py
class CountermeasureLogWidget:
    def __init__(self):
        self.logs = []

    def add_log(self, agent_id, action, result):
        self.logs.append({"agent_id": agent_id, "action": action, "result": result})

    def render(self):
        return self.logs[-10:]
