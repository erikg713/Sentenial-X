import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_model(model, vectorizer, save_path: Path, metadata: dict):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "vectorizer": vectorizer,
        "metadata": metadata
    }
    joblib.dump(payload, save_path)
    logger.info("Model and metadata saved to %s", save_path)
