# -*- coding: utf-8 -*-
"""
Cortex Module for Sentenial-X
-----------------------------

Central AI reasoning engine.
Aggregates telemetry, NLP analysis, adversarial detection,
and predictive modeling. Feeds results into simulator playbooks
or real-time alerting pipelines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ai_core import get_nlp_analyzer, get_adversarial_detector, get_predictive_model
from core.simulator import EmulationManager, TelemetryCollector

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Cortex:
    """
    Central intelligence engine for analyzing threats.
    """

    def __init__(self):
        self.nlp = get_nlp_analyzer()
        self.adversarial = get_adversarial_detector()
        self.predictive = get_predictive_model()
        self.emulation_manager = EmulationManager()
        self.telemetry_collector = TelemetryCollector()
        logger.info("Cortex initialized with AI and Simulator modules")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a text input for threats using NLP and adversarial detection.
        Returns a combined threat report.
        """
        nlp_result = self.nlp.analyze(text)
        adv_result = self.adversarial.detect(text)

        combined_threat_score = max(
            nlp_result.get("score", 0.0),
            adv_result.get("score", 0.0)
        )

        is_threat_detected = nlp_result.get("is_threat", False) or adv_result.get("is_adversarial", False)

        report = {
            "nlp": nlp_result,
            "adversarial": adv_result,
            "combined_threat_score": combined_threat_score,
            "is_threat_detected": is_threat_detected
        }

        # Collect telemetry
        self.telemetry_collector.add({
            "text": text,
            "report": report
        })

        logger.debug("Cortex text analysis report: %s", report)
        return report

    def predict_risk(self, features: List[float]) -> Dict[str, Any]:
        """
        Predict risk using predictive threat model.
        """
        predictive_result = self.predictive.predict(features)
        self.telemetry_collector.add({
            "features": features,
            "predictive_result": predictive_result
        })
        logger.debug("Cortex predictive risk report: %s", predictive_result)
        return predictive_result

    def full_analysis(self, text: str, features: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Perform a full analysis combining text and feature-based predictive modeling.
        Can trigger emulations if threat detected.
        """
        report = self.analyze_text(text)

        if features is not None:
            report["predictive"] = self.predict_risk(features)

        # Trigger emulation if threat score is high
        if report.get("is_threat_detected") or (report.get("predictive", {}).get("is_risky", False)):
            self.emulation_manager.run_all(sequential=True)
            logger.info("Threat detected: triggered simulator emulations")

        return report
