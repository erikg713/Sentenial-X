from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Tuple
import logging

class SemanticAnalyzer:
    def __init__(self, model_path: str):
        """Initialize the semantic analyzer with a pre-trained model."""
        self.logger = logging.getLogger(__name__)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.intent_labels = ["benign", "malicious", "phishing", "exploitation"]  # Example labels
            self.logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise

    def predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict the intent of the input text and return the intent and confidence score."""
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Get probabilities
            probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
            max_prob = max(probs)
            intent_idx = probs.index(max_prob)
            intent = self.intent_labels[intent_idx]
            return intent, max_prob
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    analyzer = SemanticAnalyzer("bert-base-uncased")
    text = "Execute arbitrary code to bypass security."
    intent, confidence = analyzer.predict_intent(text)
    print(f"Intent: {intent}, Confidence: {confidence:.2f}")
