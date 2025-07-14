"""
Sentenial X :: Zero-Day Behavior Detector

This module performs heuristic and statistical evaluation of unknown threat behavior
to detect potential zero-day attacks in the absence of known IOCs or signatures.

Techniques:
- Heuristic rules for suspicious behavior combinations
- Entropy analysis for obfuscated payloads
- Rare feature pattern detection
"""

import math
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("ZeroDayDetector")
logging.basicConfig(level=logging.INFO)


class ZeroDayDetector:
    def __init__(self, entropy_threshold: float = 7.0):
        """
        :param entropy_threshold: Entropy score above which payloads are considered suspicious
        """
        self.entropy_threshold = entropy_threshold

    def analyze_payload_entropy(self, data: bytes) -> float:
        """Compute Shannon entropy of a binary or text payload."""
        if not data:
            return 0.0

        byte_counts = {}
        for byte in data:
            byte_counts[byte] = byte_counts.get(byte, 0) + 1

        entropy = 0.0
        length = len(data)
        for count in byte_counts.values():
            p_x = count / length
            entropy -= p_x * math.log2(p_x)

        return entropy

    def is_obfuscated(self, payload: bytes) -> bool:
        """Returns True if the payload entropy exceeds the threshold."""
        entropy = self.analyze_payload_entropy(payload)
        logger.debug(f"Payload entropy: {entropy:.2f}")
        return entropy >= self.entropy_threshold

    def detect_rare_patterns(self, feature_vector: Dict[str, Any]) -> bool:
        """
        Heuristic checks for rare or abnormal combinations in telemetry.
        Example rules can include:
          - Uncommon port + script execution
          - Base64 + new process + remote IP
        """
        score = 0
        if feature_vector.get("used_base64", False):
            score += 1
        if feature_vector.get("remote_ip") and feature_vector.get("suspicious_port"):
            score += 1
        if feature_vector.get("process_chain_length", 0) > 5:
            score += 1
        if feature_vector.get("executed_from_temp"):
            score += 1

        logger.debug(f"Zero-day heuristic score: {score}")
        return score >= 3

    def run_analysis(
        self,
        telemetry: Dict[str, Any],
        payload: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Main zero-day detection interface.
        :param telemetry: Structured telemetry event or feature vector
        :param payload: Optional binary content (e.g., dropped file)
        :return: Detection result
        """
        result = {
            "zero_day_suspected": False,
            "reasons": [],
            "entropy_score": None
        }

        if payload:
            entropy = self.analyze_payload_entropy(payload)
            result["entropy_score"] = entropy
            if self.is_obfuscated(payload):
                result["zero_day_suspected"] = True
                result["reasons"].append("High payload entropy (possible obfuscation)")

        if self.detect_rare_patterns(telemetry):
            result["zero_day_suspected"] = True
            result["reasons"].append("Rare behavioral pattern detected")

        return result
