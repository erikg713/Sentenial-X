import logging
from datetime import datetime
from typing import Dict, List

import random
import time

logger = logging.getLogger("WormGPTClone")
logging.basicConfig(level=logging.INFO)


class WormGPTClone:
    """
    Simulated malicious LLM module for testing AI misuse and offensive prompt generation.
    Used for red teaming and adversarial simulation only.
    """

    def __init__(self, model_name: str = "wormgpt-clone-v0"):
        self.model_name = model_name
        self.loaded = False
        self._load_model()

    def _load_model(self):
        # Simulate model load
        logger.info(f"[{self.model_name}] Loading WormGPT simulation model...")
        time.sleep(1)
        self.loaded = True
        logger.info(f"[{self.model_name}] Model ready.")

    def generate_payload(self, task_type: str = "phishing") -> Dict:
        """
        Generate simulated malicious content based on task type.
        """

        if task_type == "phishing":
            return {
                "type": "phishing_email",
                "content": self._generate_phishing_email(),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        elif task_type == "macro_virus":
            return {
                "type": "malicious_macro",
                "content": self._generate_macro_code(),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        elif task_type == "ransomware_note":
            return {
                "type": "ransom_note",
                "content": self._generate_ransomware_note(),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

        else:
            logger.warning(f"Unknown task type '{task_type}', generating default payload.")
            return {
                "type": "generic",
                "content": "echo 'Simulated worm payload executed.'",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

    def _generate_phishing_email(self) -> str:
        subjects = [
            "URGENT: Your Account Has Been Compromised!",
            "Final Notice: Invoice Payment Required",
            "Security Alert: Suspicious Login Detected"
        ]
        bodies = [
            "Please log in immediately using this secure portal: http://fakebank-login.com",
            "Your email access will be suspended unless action is taken now.",
            "Verify your credentials by opening the attached document."
        ]
        return f"Subject: {random.choice(subjects)}\n\n{random.choice(bodies)}"

    def _generate_macro_code(self) -> str:
        return """Sub AutoOpen()
    Dim fso As Object
    Dim file As Object
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set file = fso.CreateTextFile("C:\\Users\\Public\\payload.bat", True)
    file.WriteLine("echo You have been pwned > C:\\Users\\Public\\readme.txt")
    file.Close
End Sub"""

    def _generate_ransomware_note(self) -> str:
        return (
            "Your files have been encrypted with AES-256.\n"
            "To recover them, send 0.1 BTC to the wallet address below:\n"
            "1HckjUpRGcrrRAtFaaCAUaGjsPx9oYmLaZ\n"
            "Failure to pay within 48 hours will result in permanent deletion."
        )


if __name__ == "__main__":
    gpt = WormGPTClone()
    payloads: List[str] = ["phishing", "macro_virus", "ransomware_note"]

    for task in payloads:
        result = gpt.generate_payload(task)
        print(f"\n--- Simulated {result['type']} ---")
        print(result["content"])
