import logging
import os

class Brainstem:
    def __init__(self):
        self.status = "idle"
        self.last_signal = None

    def process_signal(self, signal):
        self.last_signal = signal
        self.status = "processing"
        # Simulate initial brainstem reflex
        if signal.get("threat_level", 0) > 7:
            return {"reflex": "isolate_system"}
        return {"reflex": "log_and_monitor"}

    def reset(self):
        self.status = "idle"
        self.last_signal = None
