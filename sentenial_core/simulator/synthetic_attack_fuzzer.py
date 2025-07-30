# sentenial_core/simulator/synthetic_attack_fuzzer.py

import random
import string
import time
from typing import Generator, Dict, Any, Optional

from loguru import logger


class SyntheticAttackFuzzer:
    """
    Generates synthetic cybersecurity attack events for testing and simulation.
    """

    ATTACK_TYPES = [
        "ransomware",
        "phishing",
        "sql_injection",
        "xss",
        "brute_force",
        "privilege_escalation",
        "dos",
        "malware_dropper",
        "data_exfiltration",
    ]

    TARGET_SYSTEMS = [
        "host-01",
        "host-02",
        "web-server-1",
        "database-prod",
        "user-laptop-17",
        "iot-device-5",
    ]

    USERS = [
        "alice", "bob", "charlie", "eve", "mallory", "trent", "peggy"
    ]

    @staticmethod
    def random_ip() -> str:
        return ".".join(str(random.randint(1, 254)) for _ in range(4))

    @staticmethod
    def random_string(length=8) -> str:
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))

    def generate_event(self) -> Dict[str, Any]:
        attack_type = random.choice(self.ATTACK_TYPES)
        target = random.choice(self.TARGET_SYSTEMS)
        user = random.choice(self.USERS)
        timestamp = time.time()

        base_event = {
            "timestamp": timestamp,
            "attack_type": attack_type,
            "target_system": target,
            "user": user,
            "source_ip": self.random_ip(),
            "description": "",
            "payload": "",
        }

        # Generate synthetic payload and description based on attack type
        if attack_type == "ransomware":
            base_event["description"] = f"Ransomware detected encrypting files on {target}."
            base_event["payload"] = f"Encrypted files with key {self.random_string(16)}"
        elif attack_type == "phishing":
            base_event["description"] = f"Phishing email sent to user {user}."
            base_event["payload"] = f"Email subject: 'Urgent password reset request'"
        elif attack_type == "sql_injection":
            base_event["description"] = f"SQL Injection attempt on {target}."
            base_event["payload"] = f"Payload: ' OR '1'='1'; --"
        elif attack_type == "xss":
            base_event["description"] = f"Cross-site scripting attempt on web app hosted on {target}."
            base_event["payload"] = "<script>alert('xss')</script>"
        elif attack_type == "brute_force":
            base_event["description"] = f"Brute force login attempts on {target} by user {user}."
            base_event["payload"] = f"Failed login attempts: {random.randint(5, 50)}"
        elif attack_type == "privilege_escalation":
            base_event["description"] = f"Privilege escalation detected on {target} by user {user}."
            base_event["payload"] = f"Attempted sudo command with invalid privileges."
        elif attack_type == "dos":
            base_event["description"] = f"Denial of Service attack targeting {target}."
            base_event["payload"] = f"High traffic from IP {base_event['source_ip']}"
        elif attack_type == "malware_dropper":
            base_event["description"] = f"Malware dropper executed on {target}."
            base_event["payload"] = f"Dropped file {self.random_string(12)}.exe"
        elif attack_type == "data_exfiltration":
            base_event["description"] = f"Data exfiltration detected from {target}."
            base_event["payload"] = f"Uploaded {random.randint(100, 10000)} MB to remote server."

        return base_event

    def stream_events(self, delay: float = 1.0) -> Generator[Dict[str, Any], None, None]:
        """
        Yields synthetic attack events indefinitely every `delay` seconds.
        """
        while True:
            event = self.generate_event()
            logger.debug(f"Generated synthetic event: {event}")
            yield event
            time.sleep(delay)


if __name__ == "__main__":
    fuzzer = SyntheticAttackFuzzer()
    try:
        for event in fuzzer.stream_events(delay=2.0):
            print(event)
    except KeyboardInterrupt:
        print("Stopped synthetic attack fuzzer.")

