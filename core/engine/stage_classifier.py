def classify_stage(event: Dict[str, Any]) -> str:
    if "psexec" in event["command"]:
        return "Lateral Movement"
    if "whoami" in event["command"]:
        return "Reconnaissance"

