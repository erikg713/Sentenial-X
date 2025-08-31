# libs/ml/pytorch/model.py

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List
import os

# ---------------------------
# Modular Transformer-based model
# ---------------------------
class TelemetryModel(nn.Module):
    """
    Transformer-based model for telemetry/log data.
    Supports:
    - Embedding extraction (self-supervised)
    - Supervised classification
    """

    def __init__(self, base_model_name: str = "bert-base-uncased", embedding_dim: int = 128, num_classes: Optional[int] = None):
        super().__init__()
        self.base_model_name = base_model_name
        self.base = AutoModel.from_pretrained(base_model_name)
        self.embedding_dim = embedding_dim
        self.projection = nn.Linear(self.base.config.hidden_size, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes else None

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Returns:
            embeddings if labels is None
            dict(loss=..., logits=...) if labels provided
        """
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_emb = outputs[:, 0]  # CLS token
        emb = self.projection(cls_emb)
        emb = nn.functional.normalize(emb, dim=-1)

        if labels is not None and self.classifier is not None:
            logits = self.classifier(emb)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return emb

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str, map_location: Optional[str] = None):
        self.load_state_dict(torch.load(path, map_location=map_location))
        print(f"Model loaded from {path}")


# ---------------------------
# Utility for batch encoding
# ---------------------------
def encode_texts(model: TelemetryModel, texts: List[str], tokenizer_name: str = "bert-base-uncased", max_len: int = 128, device: Optional[str] = None):
    """
    Encode a list of texts into embeddings.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    embeddings = []

    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            emb = model(input_ids, attention_mask)
            embeddings.append(emb.cpu().numpy())

    return torch.vstack([torch.tensor(e) for e in embeddings]).numpy()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    sample_texts = [
        "Agent executed task A",
        "Telemetry anomaly detected",
        "Normal telemetry received"
    ]

    # Classification example
    labels = torch.tensor([0, 1, 0])
    model = TelemetryModel(base_model_name="bert-base-uncased", embedding_dim=64, num_classes=2)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    encoded = tokenizer(sample_texts, padding=True, truncation=True, return_tensors="pt")
    output = model(encoded["input_ids"], encoded["attention_mask"], labels=labels)
    print("Loss:", output["loss"].item(), "Logits:", output["logits"])

    # Embedding example
    embeddings = encode_texts(model, sample_texts)
    print("Embeddings shape:", embeddings.shape)
