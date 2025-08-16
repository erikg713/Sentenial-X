# apps/threat-engine/llm_analyzer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class LLMAnalyzer:
    """
    Large Language Model based threat predictor.
    """

    def __init__(self, model_path="models/llm/threat_classifier"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, logs, telemetry):
        """
        Predict threats from logs and telemetry using LLM.
        """
        inputs = [str(log) for log in logs]
        if not inputs:
            return []

        encodings = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)
        # Example: if class 1 > 0.5, consider as threat
        threats = []
        for i, p in enumerate(probs):
            if p[1] > 0.5:
                threats.append({
                    "type": "LLM_Predicted",
                    "severity": float(p[1]),
                    "log": logs[i]
                })
        return threats 