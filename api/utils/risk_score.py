def calculate_risk(data):
    score = 0
    if data.get("severity") == "high":
        score += 50
    if "APT29" in data.get("tags", []):
        score += 30
    if data.get("source") in ["external", "unknown"]:
        score += 20
    return min(score, 100)
