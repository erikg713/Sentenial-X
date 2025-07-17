import logging
from pathlib import Path
from typing import List, Tuple

import joblib
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_model(
    texts: List[str],
    labels: List[str],
    *,
    c: float = 1.0,
    max_df: float = 1.0,
    min_df: int = 1,
    ngram_range: Tuple[int, int] = (1, 1),
    cv: int = 5,
    save_path: Path = Path("secure_db/pipeline.pkl"),
    mlflow_experiment: str = "SentenialX_TextClassifier"
) -> Tuple[LogisticRegression, TfidfVectorizer]:
    """
    Vectorize input texts with TF-IDF and train a Logistic Regression classifier.
    Logs parameters and metrics to MLflow, and saves the pipeline to disk.
    Returns the trained model and the fitted vectorizer.
    """

    # 1. Input validation
    if not texts or not labels:
        logger.error("Empty texts or labels provided for training.")
        raise ValueError("Both `texts` and `labels` must be non-empty lists.")

    if len(texts) != len(labels):
        logger.error("Mismatch: %d texts vs %d labels.", len(texts), len(labels))
        raise ValueError("`texts` and `labels` must be the same length.")

    logger.info("Starting training on %d samples (%d classes).",
                len(texts), len(set(labels)))

    # 2. Start MLflow run
    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("C", c)
        mlflow.log_param("max_df", max_df)
        mlflow.log_param("min_df", min_df)
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("cv_folds", cv)

        # 3. Vectorization
        vectorizer = TfidfVectorizer(
            max_df=max_df,
            min_df=min_df,
            ngram_range=ngram_range
        )
        X = vectorizer.fit_transform(texts)

        # 4. Model training
        model = LogisticRegression(C=c, random_state=42, max_iter=1000)
        model.fit(X, labels)

        # 5. Cross-validation for quick metric
        scores = cross_val_score(model, X, labels, cv=cv, scoring="accuracy")
        avg_score = scores.mean()
        mlflow.log_metric("cv_accuracy", avg_score)
        logger.info("Cross-validation accuracy (cv=%d): %.4f", cv, avg_score)

        # 6. Persist pipeline
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "vectorizer": vectorizer}, save_path)
        logger.info("Pipeline saved to %s", save_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sk_model",
            registered_model_name="SentenialX_Classifier"
        )
        mlflow.log_artifact(str(save_path), artifact_path="pipeline")

    return model, vectorizer
