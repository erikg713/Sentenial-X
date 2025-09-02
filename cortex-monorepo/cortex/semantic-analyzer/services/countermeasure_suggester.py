from models.base import AnalysisResult

def suggest_countermeasures(analysis: AnalysisResult) -> dict:
    """
    Suggest automated or manual countermeasures based on AI analysis.
    """
    countermeasures = []
    if analysis.severity in ["high", "critical"]:
        countermeasures.append("Isolate host")
        countermeasures.append("Block source IP")
    elif analysis.severity == "medium":
        countermeasures.append("Trigger alert to SOC")
    else:
        countermeasures.append("Monitor only")

    return {"countermeasures": countermeasures, "risk_score": analysis.risk_score}
