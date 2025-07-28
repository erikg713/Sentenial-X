class DecisionEngine:
    def __init__(self):
        self.threat_memory = []

    def evaluate(self, signal, semantic_output):
        decision = {
            "action": "none",
            "confidence": 0.0,
            "reason": ""
        }

        if semantic_output.get("intent") == "breach":
            decision.update({
                "action": "activate_firewall_rules",
                "confidence": 0.95,
                "reason": "Detected breach intent via semantic analysis"
            })
        elif signal.get("threat_level", 0) > 8:
            decision.update({
                "action": "terminate_process",
                "confidence": 0.88,
                "reason": "High threat level based on raw signal"
            })

        self.threat_memory.append((signal, decision))
        return decision
