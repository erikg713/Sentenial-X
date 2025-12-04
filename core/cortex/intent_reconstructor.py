"""
core/cortex/intent_reconstructor.py

Intent Reconstructor
--------------------
This module estimates *user/system intent* based on signals fused by Cortex.

It performs:
- Contextual tagging
- Motivation heuristics
- Intent clustering (non-ML)
- Confidence scoring
- Behavioral interpretation

This is COMPLETELY SAFE — no offensive logic.

Output example:
    IntentResult(
        intent="configuration_change",
        confidence=0.71,
        tags=["update", "modify-settings"],
        reasons=["keywords: update, set, change"]
    )
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


# --------------------------------------------------------
# intent categories your pipeline recognises
# --------------------------------------------------------

INTENT_LABELS = {
    "query": "Information-seeking / lookup",
    "modify": "Modify or configure something",
    "system_action": "Requesting a safe system action",
    "diagnostic": "Debugging / checking status",
    "anomaly": "Unexpected or suspicious activity (non-malicious)",
    "unknown": "Unclassified"
}


# --------------------------------------------------------
# Result object
# --------------------------------------------------------

@dataclass
class IntentResult:
    intent: str
    confidence: float
    tags: List[str]
    reasons: List[str]
    metadata: Dict[str, Any]


# --------------------------------------------------------
# Intent Reconstructor
# --------------------------------------------------------

class IntentReconstructor:

    def __init__(self):
        # keyword map for safe intent inference
        self.keyword_rules = {
            "query": ["get", "show", "lookup", "fetch", "list", "explain"],
            "modify": ["set", "update", "change", "config", "modify"],
            "system_action": ["run", "start", "trigger", "execute", "build"],
            "diagnostic": ["check", "status", "inspect", "validate"],
            "anomaly": ["error", "broken", "failed", "unexpected"]
        }

    # ----------------------------------------------------
    # Keyword extraction
    # ----------------------------------------------------

    def extract_keywords(self, text: str) -> List[str]:
        words = text.lower().replace(",", " ").split()
        return [w.strip() for w in words if w.strip()]

    # ----------------------------------------------------
    # Scoring helpers
    # ----------------------------------------------------

    def score_intent(self, extracted: List[str]) -> Dict[str, float]:
        """
        Returns a score for each intent type based on keyword presence.
        """
        scores = {k: 0.0 for k in INTENT_LABELS.keys()}

        for intent, kw_list in self.keyword_rules.items():
            for kw in kw_list:
                matches = sum(1 for w in extracted if w.startswith(kw))
                scores[intent] += matches * 0.2  # each match increases confidence

        return scores

    # ----------------------------------------------------
    # Main reconstruction logic
    # ----------------------------------------------------

    def reconstruct(
        self,
        text: str,
        signals: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None
    ) -> IntentResult:
        """
        Main API.
        text  = input content
        signals = fused cortex signals (inspection/anomaly/etc.)
        meta = optional context (source info, module, channel, etc.)
        """

        signals = signals or {}
        meta = meta or {}

        extracted = self.extract_keywords(text)
        scores = self.score_intent(extracted)

        # Adjust for anomaly context (very small safe adjustment)
        if signals.get("anomaly_score_raw", 0) > 3:
            scores["anomaly"] += 0.3

        # Determine final intent
        final_intent = max(scores.items(), key=lambda x: x[1])[0]
        confidence = min(1.0, scores[final_intent])

        # Tagging
        tags = [t for t, v in scores.items() if v > 0.1]

        # Reasons
        reasons = []
        for intent, kw_list in self.keyword_rules.items():
            used = [kw for kw in kw_list if any(w.startswith(kw) for w in extracted)]
            if used:
                reasons.append(f"{intent}: keywords matched: {', '.join(used)}")

        # If absolutely no signal, fallback
        if confidence <= 0.05:
            final_intent = "unknown"
            confidence = 0.05
            reasons.append("No strong keyword or signal match")

        return IntentResult(
            intent=final_intent,
            confidence=confidence,
            tags=tags,
            reasons=reasons,
            metadata={
                "keywords": extracted,
                "raw_scores": scores,
                **meta
            }
        )


# --------------------------------------------------------
# Self-test
# --------------------------------------------------------

if __name__ == "__main__":
    ir = IntentReconstructor()

    sample_text = "Can you update the config and show me the status?"
    sample_signals = {
        "anomaly_score_raw": 0.4,
        "length": 42
    }

    result = ir.reconstruct(sample_text, sample_signals)
    print(result)
