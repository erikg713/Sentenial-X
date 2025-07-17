import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

def load_feedback(feedback_path: Path) -> Tuple[List[str], List[str]]:
    if not feedback_path.exists():
        logger.error("Feedback file does not exist: %s", feedback_path)
        raise FileNotFoundError(f"Feedback file not found: {feedback_path}")

    with open(feedback_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.error("Invalid feedback format: expected a list of dicts")
        raise ValueError("Feedback data should be a list of JSON objects")

    texts, labels = [], []
    for entry in data:
        if not isinstance(entry, dict) or "text" not in entry or "label" not in entry:
            logger.warning("Skipping malformed entry: %s", entry)
            continue
        texts.append(entry["text"])
        labels.append(entry["label"])

    if not texts:
        logger.error("No valid data found in feedback file.")
        raise ValueError("No valid feedback entries.")

    return texts, labels
