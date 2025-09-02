def calculate_risk_score(severity: str, confidence: float) -> float:
    """
    Convert AI severity and confidence into a normalized risk score [0.0, 1.0].
    """
    severity_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 0.95}
    base_score = severity_map.get(severity.lower(), 0.5)
    return min(1.0, base_score * confidence)
