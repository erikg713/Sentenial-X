from sentenialx.ai_core import detect_prompt_threat, log_threat_event

confidence = detect_prompt_threat("user input text")
if confidence > 0.85:
    log_threat_event("ai_prompt_threat", "daemon", "user input text", confidence)

