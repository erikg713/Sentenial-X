# countermeasures/trap_engine.py

import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("TrapEngine")
logger.setLevel(logging.INFO)


class Honeypot:
    def __init__(self, name: str, lure_data: Dict[str, Any]) -> None:
        self.name = name
        self.lure_data = lure_data
        self.tripped = False
        self.trip_log = []

    def is_tripped(self, access_payload: Dict[str, Any]) -> bool:
        # Simple decoy trigger: if access pattern matches, trip the honeypot
        suspicious = access_payload.get("path") in self.lure_data.get("fake_paths", [])
        if suspicious:
            self.tripped = True
            self.trip_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "payload": access_payload
            })
            logger.warning(f"Honeypot '{self.name}' tripped by {access_payload}")
        return suspicious

    def get_trip_log(self):
        return self.trip_log.copy()


class DeceptionEngine:
    def __init__(self) -> None:
        self.deceptive_responses = {
            "default": {"status": 200, "data": "Nothing to see here."}
        }

    def respond(self, access_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Respond based on payload, possibly injecting misleading information
        if access_payload.get("intent") == "enumerate":
            response = {"status": 404, "data": "Resource not found."}
        else:
            response = self.deceptive_responses["default"]
        logger.info(f"DeceptionEngine response: {response}")
        return response


class RerouteManager:
    def reroute(self, access_payload: Dict[str, Any]) -> Optional[str]:
        # Sample logic: reroute attackers to a sandbox or loopback
        if access_payload.get("threat_level", 0) > 5:
            logger.info(f"Rerouting: {access_payload} -> /sandbox")
            return "/sandbox"
        return None


class TrapEngine:
    def __init__(self) -> None:
        self.honeypots = [
            Honeypot("DecoyDB", {"fake_paths": ["/admin", "/secret-db"]}),
            Honeypot("FakeS3", {"fake_paths": ["/bucket/private", "/bucket/backup"]}),
        ]
        self.deception_engine = DeceptionEngine()
        self.rerouter = RerouteManager()

    def process_access(self, access_payload: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: Honeypot detection
        for honeypot in self.honeypots:
            if honeypot.is_tripped(access_payload):
                logger.warning(f"Trap triggered: {honeypot.name}")
                # Optionally escalate or alert here

        # Step 2: Deception response
        deception_response = self.deception_engine.respond(access_payload)

        # Step 3: Reroute if necessary
        reroute_path = self.rerouter.reroute(access_payload)
        if reroute_path:
            deception_response["reroute"] = reroute_path

        return deception_response

    def get_honeypot_logs(self):
        return {h.name: h.get_trip_log() for h in self.honeypots}


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = TrapEngine()

    # Simulated incoming access attempt
    payload = {"path": "/admin", "intent": "scan", "threat_level": 8}
    result = engine.process_access(payload)
    print("TrapEngine result:", result)
    print("Honeypot logs:", engine.get_honeypot_logs())
