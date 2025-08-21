# apps/dashboard/pages/widgets/network_traffic.py
class NetworkTrafficWidget:
    def __init__(self):
        self.traffic_logs = []

    def add_log(self, agent_id, bytes_sent, bytes_recv):
        self.traffic_logs.append({
            "agent_id": agent_id,
            "bytes_sent": bytes_sent,
            "bytes_recv": bytes_recv
        })

    def render(self):
        return self.traffic_logs[-20:]
