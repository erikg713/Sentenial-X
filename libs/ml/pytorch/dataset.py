# libs/ml/pytorch/dataset.py

from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from typing import List, Optional


# ---------------------------
# Supervised dataset
# ---------------------------
class TelemetryDataset(Dataset):
    """
    PyTorch dataset for supervised telemetry/log classification.
    """
    def __init__(self, texts: List[str], labels: List[int], tokenizer_name: str = "bert-base-uncased", max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------
# Contrastive / self-supervised dataset
# ---------------------------
class TelemetryContrastiveDataset(Dataset):
    """
    Dataset returning two augmented views of the same telemetry/log text
    for contrastive learning.
    """
    def __init__(self, texts: List[str], tokenizer_name: str = "bert-base-uncased", max_len: int = 128):
        self.texts = texts
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Two views of the same text
        view1 = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        view2 = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        # Squeeze batch dimension
        v1 = {k: v.squeeze(0) for k, v in view1.items()}
        v2 = {k: v.squeeze(0) for k, v in view2.items()}
        return v1, v2


# ---------------------------
# Utility functions
# ---------------------------
def get_supervised_dataset(texts: List[str], labels: List[int], tokenizer_name="bert-base-uncased", max_len=128):
    return TelemetryDataset(texts, labels, tokenizer_name=tokenizer_name, max_len=max_len)


def get_contrastive_dataset(texts: List[str], tokenizer_name="bert-base-uncased", max_len=128):
    return TelemetryContrastiveDataset(texts, tokenizer_name=tokenizer_name, max_len=max_len)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    sample_texts = ["Agent executed task A", "Telemetry anomaly detected", "Normal telemetry received"]
    sample_labels = [0, 1, 0]

    supervised_dataset = get_supervised_dataset(sample_texts, sample_labels)
    print("Supervised sample:", supervised_dataset[0])

    contrastive_dataset = get_contrastive_dataset(sample_texts)
    v1, v2 = contrastive_dataset[0]
    print("Contrastive view1:", v1)
    print("Contrastive view2:", v2)
