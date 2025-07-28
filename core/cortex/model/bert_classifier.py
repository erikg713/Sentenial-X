from transformers import pipeline
import torch

class BERTThreatClassifier:
    def __init__(self):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline(
            "text-classification",
            model="bhadresh-savani/bert-base-uncased-emotion",  # placeholder, see trainer below
            device=device
        )

        self.intent_map = {
            "fear": "breach",
            "anger": "exploit",
            "surprise": "malware",
            "joy": "benign",
            "love": "benign",
            "sadness": "unknown"
        }

    def classify(self, text: str):
        result = self.classifier(text, truncation=True)[0]
        label = result['label'].lower()
        score = result['score']
        return self.intent_map.get(label, "unknown"), score
