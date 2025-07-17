import os import json import logging import threading from datetime import datetime from typing import Dict, Any, Optional

from ai_core.language_model import NLPClassifier from ai_core.threat_ranker import ThreatRanker from orchestrator.response_handler import ResponseHandler from engine.incident_logger import IncidentLogger

class Cortex: """ The central decision-making module of Sentenial-X. It processes threat inputs, evaluates severity, and dispatches countermeasures. """

def __init__(self):
    self.logger = logging.getLogger("SentenialX.Cortex")
    self.nlp = NLPClassifier()
    self.ranker = ThreatRanker()
    self.responder = ResponseHandler()
    self.incident_logger = IncidentLogger()
    self.lock = threading.Lock()

def process_threat_event(self, threat_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entry point for threat data ingestion from engine modules.
    Args:
        threat_report (dict): structured event report
    Returns:
        dict: response summary
    """
    try:
        self.logger.info("[CORTEX] Received threat report")
        threat_id = threat_report.get("id", f"event-{datetime.utcnow().isoformat()}")
        
        # NLP-based threat classification
        description = threat_report.get("description", "")
        category, confidence = self.nlp.classify(description)
        
        # Rank threat severity
        threat_score = self.ranker.score(threat_report, category)

        self.logger.debug(f"Threat '{threat_id}' classified as '{category}' with score {threat_score}")

        # Log the threat
        self.incident_logger.log(threat_id, threat_report, category, threat_score)

        # Decide on action
        if threat_score >= 75:
            action_result = self.responder.deploy_countermeasure(threat_report)
            status = "countermeasure-deployed"
        elif threat_score >= 50:
            action_result = self.responder.isolate_source(threat_report)
            status = "source-isolated"
        else

