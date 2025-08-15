# sentenial-x/ai_core/threat_classifier.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
from .config import MODEL_PATHS, DEVICE, THREAT_SCORE_THRESHOLD

class ThreatClassifier:
    """
    BERT/LoRA-based threat classifier.
    Predicts threat labels and confidence scores for logs.
    """

    def __init__(self):
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS["bert_threat_classifier"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATHS["bert_threat_classifier"]
        ).to(self.device)
        self.model.eval()
        self.labels = ["normal", "malware", "sql_injection", "xss"]

    @torch.no_grad()
    def predict(self, logs: List[str]) -> Tuple[List[str], List[float]]:
        encodings = self.tokenizer(
            logs, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1)
        scores, indices = torch.max(probs, dim=-1)
        labels = [self.labels[idx] for idx in indices.tolist()]
        # Apply threshold
        labels = [label if score >= THREAT_SCORE_THRESHOLD else "normal"
                  for label, score in zip(labels, scores.tolist())]
        return labels, scores.tolist()
