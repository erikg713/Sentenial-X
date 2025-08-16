# apps/ml_pipeline/train_classifier.py
import os
import pickle
import json
import aiosqlite
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from config import DB_PATH, MODEL_PATH
from logger import setup_logger

logger = setup_logger("ml_classifier")

class NaiveBayesClassifier:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()

    async def load_feedback(self, db_path=DB_PATH):
        """
        Load labeled feedback data from SQLite asynchronously.
        Returns:
            texts (list[str]), labels (list[int])
        """
        if not os.path.exists(db_path):
            logger.warning(f"Database {db_path} not found.")
            return [], []

        texts, labels = [], []
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT text, label FROM feedback") as cursor:
                async for row in cursor:
                    texts.append(row[0])
                    labels.append(row[1])
        logger.info(f"Loaded {len(texts)} feedback entries from DB.")
        return texts, labels

    def train(self, texts, labels):
        """
        Fit the vectorizer and Naive Bayes model.
        """
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        logger.info("Training completed.")

    def predict(self, texts):
        """
        Predict labels for new texts.
        """
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def save(self, path=MODEL_PATH):
        """
        Save the trained model and vectorizer.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "model": self.model
            }, f)
        logger.info(f"Model saved to {path}")

    @staticmethod
    def load(path=MODEL_PATH):
        """
        Load a previously trained model and vectorizer.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} does not exist")
        with open(path, "rb") as f:
            data = pickle.load(f)
        classifier = NaiveBayesClassifier()
        classifier.vectorizer = data["vectorizer"]
        classifier.model = data["model"]
        logger.info(f"Model loaded from {path}")
        return classifier

# -------------------------
# Example Usage
# -------------------------
import asyncio

async def main():
    nb = NaiveBayesClassifier()
    texts, labels = await nb.load_feedback()
    if texts:
        nb.train(texts, labels)
        nb.save()
    else:
        logger.warning("No data to train on.")

if __name__ == "__main__":
    asyncio.run(main())