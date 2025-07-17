from transformers import pipeline

_model = None

def load_local_model():
    global _model
    if _model is None:
        _model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    return _model