# src/encoder_training.py

import os
from typing import List, Optional
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# ---------------------------
# Dataset
# ---------------------------
class TelemetryDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer_name: str = "bert-base-uncased", max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ---------------------------
# Transformer Encoder
# ---------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", embedding_dim=256):
        super().__init__()
        self.base = AutoModel.from_pretrained(base_model_name)
        self.proj = nn.Linear(self.base.config.hidden_size, embedding_dim)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = outputs[:, 0]  # CLS token
        embeddings = self.proj(cls)
        return embeddings

# ---------------------------
# Training function
# ---------------------------
def train_encoder(
    texts: List[str],
    labels: Optional[List[int]] = None,
    model_name: str = "bert-base-uncased",
    embedding_dim: int = 256,
    batch_size: int = 16,
    epochs: int = 3,
    lr: float = 5e-5,
    save_path: str = "models/encoder.pt"
):
    # Split dataset
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42) if labels else (texts, texts, labels, labels)
    
    train_dataset = TelemetryDataset(train_texts, train_labels, tokenizer_name=model_name)
    val_dataset = TelemetryDataset(val_texts, val_labels, tokenizer_name=model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerEncoder(base_model_name=model_name, embedding_dim=embedding_dim).to(device)

    # HuggingFace Trainer boilerplate
    class HFWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
            self.classifier = nn.Linear(embedding_dim, 2) if labels else None

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            emb = self.encoder(input_ids, attention_mask)
            if labels is not None:
                logits = self.classifier(emb)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                return {"loss": loss, "logits": logits}
            return {"embeddings": emb}

    wrapped_model = HFWrapper(model)

    training_args = TrainingArguments(
        output_dir="./training_output",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if labels else "no",
        report_to="none"
    )

    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if labels else None
    )

    trainer.train()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Encoder saved to {save_path}")
    return model

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    sample_texts = [
        "Agent executed task A",
        "Telemetry event received",
        "Malicious activity detected"
    ]
    sample_labels = [0, 0, 1]  # 0=benign, 1=malicious
    model = train_encoder(sample_texts, labels=sample_labels, embedding_dim=128, epochs=1)
