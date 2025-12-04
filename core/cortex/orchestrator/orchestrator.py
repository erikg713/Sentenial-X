#!/usr/bin/env python3
"""
Sentenial-X :: Cortex Orchestrator
==================================

Purpose:
    Central orchestrator for the Cortex AI stack:
        - Accepts raw signals/events
        - Executes semantic analysis
        - Runs zero-day prediction
        - Runs NLP / BERT threat intent classification
        - Computes final risk scores
        - Outputs structured threat intelligence
"""

import json
from typing import Dict, Any, List

from core.cortex.zero_day_predictor import ZeroDayPredictor
from core.cortex.models import NLPModel, BertClassifier
from core.cortex.datasets import Threat_intents  # assuming loader exists
from core.cortex.semantic_analyzer.parser import SemanticAnalyzer  # adjust import if needed


# ------------------------------------------------------------
# Orchestrator Class
# ------------------------------------------------------------
class CortexOrchestrator:
    def __init__(
        self,
        zero_day_predictor: ZeroDayPredictor = None,
        nlp_model: NLPModel = None,
        bert_model: BertClassifier = None,
        semantic_analyzer: SemanticAnalyzer = None,
        threat_intents: List[Dict[str, Any]] = None
    ):
        self.zero_day = zero_day_predictor or ZeroDayPredictor()
        self.nlp_model = nlp_model or NLPModel(backend="tfidf")
        self.bert_model = bert_model or BertClassifier(num_labels=2)
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer()
        self.threat_intents = threat_intents or self._load_default_threat_intents()

    # --------------------------------------------------------
    # Default Threat Intents Loader
    # --------------------------------------------------------
    def _load_default_threat_intents(self):
        try:
            import csv
            intents = []
            with open("core/cortex/datasets/Threat_intents.csv", "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert numeric fields
                    row['confidence'] = float(row['confidence'])
                    intents.append(row)
            return intents
        except FileNotFoundError:
            print("[WARNING] Threat_intents.csv not found. Continuing without.")
            return []

    # --------------------------------------------------------
    # Process a single signal/event
    # --------------------------------------------------------
    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        # 1️⃣ Semantic Analysis
        semantic_result = self.semantic_analyzer.analyze(event)

        # 2️⃣ Zero-Day Prediction
        zero_day_result = self.zero_day.analyze(event)

        # 3️⃣ NLP Model Classification
        text_features = [event.get("log_text", "")]  # assume log_text exists
        nlp_probs = self.nlp_model.predict_proba(text_features)

        # 4️⃣ BERT Classification (optional)
        bert_probs = self.bert_model.predict(text_features)

        # 5️⃣ Match threat intents
        matched_intents = self._match_threat_intents(event, nlp_probs)

        # 6️⃣ Aggregate results
        aggregated = {
            "event_id": event.get("event_id"),
            "semantic": semantic_result,
            "zero_day": zero_day_result,
            "nlp_probs": nlp_probs,
            "bert_probs": bert_probs,
            "matched_intents": matched_intents
        }
        return aggregated

    # --------------------------------------------------------
    # Match NLP probabilities to Threat_intents.csv
    # --------------------------------------------------------
    def _match_threat_intents(self, event: Dict[str, Any], nlp_probs: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Simple matching: selects top intents based on NLP probability
        """
        if not self.threat_intents or not nlp_probs:
            return []

        top_prob = nlp_probs[0]  # single event
        sorted_probs = sorted(top_prob.items(), key=lambda x: x[1], reverse=True)

        matched = []
        for i, prob in sorted_probs[:3]:  # top 3
            intent_id = i
            # map intent_id to dataset entry if possible
            intent_data = next((t for t in self.threat_intents if t["intent_id"] == f"T{int(i)+1:03}"), None)
            if intent_data:
                matched.append({
                    "intent_id": intent_data["intent_id"],
                    "threat_intent": intent_data["threat_intent"],
                    "probability": prob,
                    "severity": intent_data["severity"]
                })
        return matched

    # --------------------------------------------------------
    # Batch Processing
    # --------------------------------------------------------
    def process_batch(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [self.process_event(event) for event in events]


# ------------------------------------------------------------
# CLI Interface
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cortex Orchestrator CLI")
    parser.add_argument("--input", required=True, help="JSON file containing events")
    parser.add_argument("--output", required=False, help="Output JSON file path")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        events = json.load(f)

    orchestrator = CortexOrchestrator()
    results = orchestrator.process_batch(events)

    output_json = json.dumps(results, indent=4)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Results saved to {args.output}")
    else:
        print(output_json)
