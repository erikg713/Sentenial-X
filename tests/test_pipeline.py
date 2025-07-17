### tests/test_pipeline.py
import json
import pytest
from pathlib import Path
from ml_pipeline.data_loader import load_feedback
from ml_pipeline.model_trainer import train_model

@pytest.fixture
def tmp_feedback_file(tmp_path):
    file = tmp_path / "feedback.json"
    feedback_data = [
        {"text": "good product", "label": "positive"},
        {"text": "bad service", "label": "negative"}
    ]
    file.write_text(json.dumps(feedback_data))
    return file

def test_load_feedback(tmp_feedback_file):
    texts, labels = load_feedback(tmp_feedback_file)
    assert len(texts) == 2
    assert labels == ["positive", "negative"]

def test_train_model(tmp_feedback_file):
    texts, labels = load_feedback(tmp_feedback_file)
    model, vectorizer = train_model(texts, labels)
    assert hasattr(model, "predict")
    assert hasattr(vectorizer, "transform")
