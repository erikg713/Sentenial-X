# sentenial-x/ai_core/countermeasure_predictor.py
from typing import List
from ..agents.config import DEFAULT_COUNTERMEASURES

class CountermeasurePredictor:
    """
    Suggests countermeasures based on threat labels and scores.
    Can be upgraded to ML-based predictive actions.
    """

    def predict(self, logs: List[str], scores: List[float]) -> List[str]:
        actions = []
        for log, score in zip(logs, scores):
            if "malware" in log.lower():
                actions.append(DEFAULT_COUNTERMEASURES["malware"])
            elif "drop table" in log.lower() or "sql injection" in log.lower():
                actions.append(DEFAULT_COUNTERMEASURES["sql_injection"])
            elif "<script>" in log.lower() or "xss" in log.lower():
                actions.append(DEFAULT_COUNTERMEASURES["xss"])
            else:
                actions.append(DEFAULT_COUNTERMEASURES["normal"])
        return actions
