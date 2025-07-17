import logging
from pathlib import Path
import json

logger = logging.getLogger("SentenialX.Feedback")
FEEDBACK_FILE = Path("secure_db/feedback_log.jsonl")

def update_model(input_text: str, label: str, source: str):
    """
    Log user or system feedback for future model fine-tuning.
    """
    feedback_entry = {
        "text": input_text,
        "label": label,
        "source": source,
        "timestamp": datetime.utcnow().isoformat()
    }
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_entry) + "\n")
    logger.info(f"Logged feedback for model update from {source}")



from sentenialx.ai_core import update_model

update_model(
    input_text="Confirmed phishing prompt.",
    label="malicious",
    source="user_review"
)

