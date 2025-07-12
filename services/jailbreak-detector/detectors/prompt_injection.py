from config import KNOWN_TRIGGERS

def detect_injection(prompt: str):
    triggers = []
    for phrase in KNOWN_TRIGGERS:
        if phrase.lower() in prompt.lower():
            triggers.append(phrase)
    return triggers
