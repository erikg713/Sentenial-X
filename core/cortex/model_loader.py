# sentenial_x/core/cortex/model_loader.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from .config import CONFIG

class CyberIntentModel:
    def __init__(self):
        self.device = torch.device(CONFIG["model"]["device"])
        self.model_path = CONFIG["model"]["custom_model_path"]
        self.max_len = CONFIG["model"]["max_seq_length"]

        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> str:
        tokens = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()

        return str(predicted_class)

