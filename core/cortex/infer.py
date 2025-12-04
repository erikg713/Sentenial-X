"""
core/cortex/infer.py

CORTEX: Inference Engine
------------------------
This module performs:
- Rule-based inference
- Signal fusion (combining results from scanners, parsers, defender, payloads)
- Confidence scoring
- Policy evaluation
- Decision routing for higher-level automation

This engine is intentionally safe and non-offensive.
It does NOT generate exploits, probe targets, or automate attacks.
It only interprets signals already produced by your authorized components.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


# ----------------------------------------------------
# Dataclasses: Unified inference output
# ----------------------------------------------------

@dataclass
class InferenceResult:
    verdict: str
    confidence: float
    reasons: List[str]
    metadata: Dict[str, Any]


# ----------------------------------------------------
# Internal scoring helpers
# ----------------------------------------------------

def normalize_score(value: float, min_v: float, max_v: float) -> float:
    """Normalize value into 0-1 range."""
    if max_v == min_v:
        return 0.0
    return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))


# ----------------------------------------------------
# Rule Engine
# ----------------------------------------------------

class RuleEngine:
    """
    Simple rule-based inference engine.

    You provide a dictionary of rules:
    {
        "rule_name": {
            "when": lambda signals: bool,
            "score": float,
            "reason": "text"
        }
    }
    """

    def __init__(self, rules: Optional[Dict[str, Dict[str, Any]]] = None):
        self.rules = rules or {}

    def add_rule(self, name: str, when, score: float, reason: str):
        self.rules[name] = {
            "when": when,
            "score": score,
            "reason": reason
        }

    def evaluate(self, signals: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Evaluate all rules and return (score_sum, reasons_triggered).
        """
        total = 0.0
        reasons = []

        for name, rule in self.rules.items():
            try:
                if rule["when"](signals):
                    total += rule["score"]
                    reasons.append(rule["reason"])
            except Exception:
                # Rule failures should not break engine
                continue

        return total, reasons


# ----------------------------------------------------
# Cortex Inference Engine
# ----------------------------------------------------

class CortexInfer:
    """
    Main inference layer.

    Combines:
    - Basic heuristics
    - Defender anomaly signals
    - Custom rule engine
    """

    def __init__(self):
        # Base rule engine
        self.re = RuleEngine()

        # Default example rules
        self._load_default_rules()

    # ---------------------------
    # Default rules
    # ---------------------------

    def _load_default_rules(self):
        self.re.add_rule(
            "long_content_flag",
            lambda s: s.get("length", 0) > 50000,
            score=0.25,
            reason="Content unusually long"
        )

        self.re.add_rule(
            "low_entropy_flag",
            lambda s: s.get("entropy", 0) < 0.5,
            score=0.1,
            reason="Low-entropy repetitive content"
        )

        self.re.add_rule(
            "blacklist_triggered",
            lambda s: s.get("blacklisted", False),
            score=0.5,
            reason="Blacklisted pattern detected"
        )

        self.re.add_rule(
            "unique_ratio_low",
            lambda s: s.get("unique_ratio", 1) < 0.1,
            score=0.2,
            reason="Content shows atypically low character diversity"
        )

    # ---------------------------
    # Fusion logic
    # ---------------------------

    def fuse_signals(
        self,
        inspection: Optional[Dict[str, Any]] = None,
        anomaly: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Combine signals from diverse modules.

        Example inputs:
            inspection = {
                "safe": True/False,
                "reasons": [...],
                "length": int,
                "hash": "...",
            }

            anomaly = {
                "score": float,
                "signals": {...}
            }
        """

        fused = {}

        # Inspection signals
        if inspection:
            fused.update({
                "length": inspection.get("length"),
                "inspection_safe": inspection.get("safe"),
                "inspection_reasons": inspection.get("reasons", []),
                "blacklisted": any("blacklisted" in r.lower() for r in inspection.get("reasons", []))
            })

        # Anomaly signals
        if anomaly:
            an_sig = anomaly.get("signals", {})
            fused.update({
                "anomaly_score_raw": anomaly.get("score", 0.0),
                **an_sig
            })

        # Meta (optional)
        if meta:
            fused.update({
                f"meta_{k}": v for k, v in meta.items()
            })

        return fused

    # ---------------------------
    # Final inference decision
    # ---------------------------

    def infer(self, signals: Dict[str, Any]) -> InferenceResult:
        """
        Convert signal dictionary → final classification.
        """

        rule_score, reasons = self.re.evaluate(signals)
        anomaly_raw = signals.get("anomaly_score_raw", 0.0)

        # Normalize anomaly score (0–1)
        anomaly_score = normalize_score(anomaly_raw, 0, 5)

        # Final combined score (bounded 0–1)
        combined = min(1.0, rule_score + anomaly_score)

        # Classification logic
        if combined < 0.2:
            verdict = "clean"
        elif combined < 0.5:
            verdict = "suspect"
        else:
            verdict = "alert"

        return InferenceResult(
            verdict=verdict,
            confidence=combined,
            reasons=reasons,
            metadata=signals
        )


# ----------------------------------------------------
# Self-test
# ----------------------------------------------------

if __name__ == "__main__":
    cortex = CortexInfer()

    # Example simulated signals
    sample_inspection = {
        "safe": False,
        "reasons": ["Content extremely long", "Blacklisted pattern detected"],
        "length": 120000
    }

    sample_anomaly = {
        "score": 3.0,
        "signals": {
            "entropy": 0.3,
            "unique_ratio": 0.05
        }
    }

    fused = cortex.fuse_signals(sample_inspection, sample_anomaly)
    result = cortex.infer(fused)

    print("Fused signals:", fused)
    print("Final decision:", result)
