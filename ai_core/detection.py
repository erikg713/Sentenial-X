# sentenialx/ai_core/detection.py

from transformers import pipeline
import logging

logger = logging.getLogger("SentenialX.Detection")

# Load zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

THREAT_LABELS = [
    "phishing",
    "malware",
    "ransomware",
    "social engineering",
    "fraud",
    "safe"
]

def detect_prompt_threat(text: str) -> float:
    """
    Analyze a given text input and return confidence score for malicious intent.
    """
    try:
        result = classifier(text, THREAT_LABELS)
        scores = dict(zip(result['labels'], result['scores']))
        threat_score = sum([scores[label] for label in THREAT_LABELS if label != "safe"])
        logger.info(f"Scored input threat: {threat_score:.2f}")
        return threat_score
    except Exception as e:
        logger.error(f"Threat detection failed: {e}")
        return 0.0


from sentenialx.ai_core import detect_prompt_threat, log_threat_event

confidence = detect_prompt_threat("user input text")
if confidence > 0.85:
    log_threat_event("ai_prompt_threat", "daemon", "user input text", confidence)

