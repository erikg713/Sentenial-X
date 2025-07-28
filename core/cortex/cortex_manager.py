"""
Sentenial X :: Cortex Manager

Main interface for executing coordinated behavioral analysis and threat classification.
Integrates all engines: semantics, profiling, anomaly detection, correlation, story building.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from core.engine.semantics_analyzer import SemanticsAnalyzer
from core.engine.correlation_engine import correlate_events
from core.engine.execution_profiler import profile_execution_chain
from core.engine.signal_enricher import enrich_event
from core.engine.stage_classifier import classify_stage
from core.engine.evasion_scoring import score_evasion
from core.engine.threat_story_builder import build_story
from core.detection.zero_day_detector import ZeroDayDetector
from core.cortex.sae_anomaly import SAEAnomalyDetector

logger = logging.getLogger("CortexManager")
logging.basicConfig(level=logging.INFO)


class CortexManager:
    def __init__(self):
        self.semantics = SemanticsAnalyzer()
        self.zeroday = ZeroDayDetector()
        self.anomaly = SAEAnomalyDetector(use_model=False)

    def analyze(
        self,
        session_id: str,
        telemetry_stream: List[Dict[str, Any]],
        payloads: Optional[List[bytes]] = None
    ) -> Dict[str, Any]:
        """
        Full behavioral analysis pipeline for a given session.
        :param session_id: Unique session ID
        :param telemetry_stream: List of telemetry events (dicts)
        :param payloads: Optional list of associated payloads (binary)
        :return: Analysis result dictionary
        """
        logger.info(f"Analyzing session: {session_id}")
        result = {
            "session_id": session_id,
            "correlated_chains": [],
            "anomalies": [],
            "zero_day_hits": [],
            "threat_stories": [],
            "telemetry_profile": {},
            "semantic_intents": []
        }

        # Normalize timestamps
        for evt in telemetry_stream:
            if isinstance(evt.get("timestamp"), str):
                evt["timestamp"] = datetime.fromisoformat(evt["timestamp"])

        # Enrich telemetry
        enriched_stream = [enrich_event(evt) for evt in telemetry_stream]

        # Correlate
        chains = correlate_events(enriched_stream)
        result["correlated_chains"] = len(chains)

        for chain in chains:
            # Stage Classification
            for evt in chain:
                evt["stage"] = classify_stage(evt)
                evt["evasion_score"] = score_evasion(evt)

            # Semantic analysis
            for evt in chain:
                if "command" in evt:
                    semantic = self.semantics.analyze_command(evt["command"])
                    evt["intents"] = semantic["intents"]
                    if semantic["intents"]:
                        result["semantic_intents"].extend(semantic["intents"])

            # Execution Profile
            profile = profile_execution_chain(chain)
            result["telemetry_profile"][chain[0].get("session_id", "unknown")] = profile

            # Threat Story
            story = build_story(chain)
            result["threat_stories"].append(story)

            # Anomaly detection (on numeric features if available)
            numeric_features = [[len(evt.get("command", "")), evt.get("evasion_score", 0.0)] for evt in chain]
            for i, vec in enumerate(numeric_features):
                hist = numeric_features[:i]
                anomaly_result = self.anomaly.detect(vec, history=hist)
                if anomaly_result["anomaly"]:
                    result["anomalies"].append({
                        "event_index": i,
                        "chain_session": chain[0].get("session_id", "unknown"),
                        "score": anomaly_result.get("score", 0.0),
                        "method": anomaly_result["method"]
                    })

        # Zero-Day Analysis
        if payloads:
            for p in payloads:
                zd = self.zeroday.run_analysis({}, payload=p)
                if zd["zero_day_suspected"]:
                    result["zero_day_hits"].append(zd)

        result["semantic_intents"] = list(set(result["semantic_intents"]))
        return result
