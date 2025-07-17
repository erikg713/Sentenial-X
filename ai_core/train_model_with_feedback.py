import json
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path("secure_db/model.pkl")

def train_model_with_feedback(feedback_path: Path):
    if not feedback_path.exists():
        logger.warning("No feedback found for training at %s", feedback_path)
        return

    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            logger.warning("Feedback file is empty.")
            return

        texts = [entry["text"] for entry in data]
        labels = [entry["label"] for entry in data]

        if not texts or not labels:
            logger.error("No valid data found for training.")
            return

        # Vectorize text
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        # Train model
        model = LogisticRegression()
        model.fit(X, labels)

        # Save model and vectorizer
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "vectorizer": vectorizer}, MODEL_PATH)

        logger.info("Model retrained and saved to %s", MODEL_PATH)

    except Exception as e:
        logger.exception("An error occurred while training the model: %s", e)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrain model with feedback data.")
    parser.add_argument("feedback_file", type=str, help="Path to the feedback JSON file.")
    args = parser.parse_args()

    feedback_path = Path(args.feedback_file)
    train_model_with_feedback(feedback_path)
