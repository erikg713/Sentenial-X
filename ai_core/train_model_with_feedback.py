from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib  # for saving model
import json
from pathlib import Path

MODEL_PATH = Path("secure_db/model.pkl")

def train_model_with_feedback(feedback_path: Path):
    if not feedback_path.exists():
        logger.warning("No feedback found for training.")
        return

    with open(feedback_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [entry["text"] for entry in data]
    labels = [entry["label"] for entry in data]

    # Vectorize text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)

    # Train model
    model = LogisticRegression()
    model.fit(X, labels)

    # Save model and vectorizer
    joblib.dump({"model": model, "vectorizer": vectorizer}, MODEL_PATH)
    logger.info("Model retrained and saved.")
