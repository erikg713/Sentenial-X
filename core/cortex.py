# -*- coding: utf-8 -*-
"""
Cortex Module for Sentenial-X
-----------------------------

Central AI reasoning engine.
Aggregates:
- NLP analysis (TextAnalyzer)
- Adversarial detection
- Predictive modeling
- Anomaly detection
Feeds results into simulator playbooks or telemetry pipelines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ai_core import get_adversarial_detector, get_predictive_model
from core.simulator import EmulationManager, TelemetryCollector
from cortex.semantic_analyzer import TextAnalyzer, AnomalyDetector

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Cortex:
    """
    Central intelligence engine for analyzing threats.
    """

    def __init__(self):
        # Core AI modules
        self.text_analyzer = TextAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.adversarial = get_adversarial_detector()
        self.predictive = get_predictive_model()

        # Simulator and telemetry
        self.emulation_manager = EmulationManager()
        self.telemetry_collector = TelemetryCollector()

        logger.info("Cortex initialized with semantic-analyzer, ai_core, and simulator modules")

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze a text input for threats using TextAnalyzer and AdversarialDetector.
        Returns a combined report.
        """
        text_result = self.text_analyzer.analyze(text)
        adv_result = self.adversarial.detect(text)

        combined_score = max(
            text_result.get("score", 0.0),
            adv_result.get("score", 0.0)
        )
        is_threat_detected = text_result.get("is_threat", False) or adv_result.get("is_adversarial", False)

        report = {
            "text_analysis": text_result,
            "adversarial": adv_result,
            "combined_threat_score": combined_score,
            "is_threat_detected": is_threat_detected
        }

        self.telemetry_collector.add({
            "text": text,
            "report": report
        })

        logger.debug("Cortex text analysis report: %s", report)
        return report

    def analyze_features(self, features: List[float]) -> Dict[str, Any]:
        """
        Analyze numeric/log features for predictive risk and anomalies.
        """
        # Predictive modeling
        predictive_result = self.predictive.predict(features)

        # Anomaly detection
        anomaly_result = self.anomaly_detector.analyze(features)

        report = {
            "predictive": predictive_result,
            "anomaly": anomaly_result
        }

        self.telemetry_collector.add({
            "features": features,
            "analysis": report
        })

        logger.debug("Cortex feature analysis report: %s", report)
        return report

    def full_analysis(self, text: str, features: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Perform full analysis combining:
        - NLP & adversarial detection on text
        - Predictive modeling & anomaly detection on features
        Can trigger simulator emulations if threat detected.
        """
        report = self.analyze_text(text)

        if features:
            report["feature_analysis"] = self.analyze_features(features)

        threat_detected = report.get("is_threat_detected", False)
        predictive_risk = report.get("feature_analysis", {}).get("predictive", {}).get("is_risky", False)
        anomaly_detected = report.get("feature_analysis", {}).get("anomaly", {}).get("is_anomaly", False)

        if threat_detected or predictive_risk or anomaly_detected:
            self.emulation_manager.run_all(sequential=True)
            logger.info("High-risk detected: triggered simulator emulations")

        return report
