def apply_policy(triggers: list) -> str:
    if not triggers:
        return "allow"
    return "blocked" if "ignore all previous instructions" in triggers else "flagged"
