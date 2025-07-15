def score_evasion(event: Dict[str, Any]) -> float:
    score = 0
    if "bypass" in event["command"]:
        score += 1.0
    if "vssadmin delete" in event["command"]:
        score += 1.5
    return score / 3.0  # normalize

