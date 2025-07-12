from detectors.prompt_injection import detect_injection
from detectors.rewriting_guard import apply_policy
from config import RISK_THRESHOLDS

def detect_prompt(prompt: str):
    triggers = detect_injection(prompt)
    score = min(1.0, 0.2 + len(triggers) * 0.25)
    risk = (
        "low" if score < RISK_THRESHOLDS["low"]
        else "medium" if score < RISK_THRESHOLDS["medium"]
        else "high"
    )
    action = apply_policy(triggers)

    return {
        "prompt": prompt,
        "score": round(score, 2),
        "risk": risk,
        "trigger": triggers or ["none"],
        "action": action
    }
