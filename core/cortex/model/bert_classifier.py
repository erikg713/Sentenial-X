from transformers import pipeline

class BERTThreatClassifier:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")  # Replace with fine-tuned threat model if available
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

