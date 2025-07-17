def save_model(model, vectorizer, save_path: Path, metadata: dict):
    data = {
        "model": model,
        "vectorizer": vectorizer,
        "metadata": metadata
    }
    joblib.dump(data, save_path)

