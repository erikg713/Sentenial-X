# libs/ml/ml_pipeline.py

from libs.ml.pytorch import (
    TelemetryModel,
    TrainerModule,
    get_supervised_dataset,
    get_contrastive_dataset,
    get_device,
    log
)
from libs.ml.pytorch.trainer import nt_xent_loss
from libs.ml.pytorch.dataset import TelemetryDataset, TelemetryContrastiveDataset
from libs.ml.pytorch.utils import batch_encode_texts, ensure_dir
from libs.ml.pytorch.lora_tuner import LoRATuner
from torch.utils.data import DataLoader
import torch
from typing import List, Optional


class SentenialMLPipeline:
    """
    Complete ML pipeline for telemetry/log analysis in Sentenial-X.
    Supports:
        - Supervised classification
        - Contrastive self-supervised embedding training
        - LoRA fine-tuning for efficiency
        - Batch inference
    """

    def __init__(self, device: Optional[str] = None):
        self.device = device or get_device()
        self.model: Optional[TelemetryModel] = None
        self.trainer: Optional[TrainerModule] = None
        self.lora_tuner: Optional[LoRATuner] = None

    # ---------------------------
    # Initialize model
    # ---------------------------
    def init_model(self, base_model_name="bert-base-uncased", embedding_dim=128, num_classes: Optional[int] = None):
        self.model = TelemetryModel(base_model_name=base_model_name, embedding_dim=embedding_dim, num_classes=num_classes)
        self.trainer = TrainerModule(self.model, device=self.device)
        log(f"Initialized model with embedding_dim={embedding_dim}, num_classes={num_classes}")

    # ---------------------------
    # Supervised training
    # ---------------------------
    def train_supervised(self, texts: List[str], labels: List[int], batch_size=16, epochs=3, lr=5e-5):
        if not self.model or not self.trainer:
            raise ValueError("Model not initialized. Call init_model() first.")
        self.trainer.train_supervised(texts, labels, batch_size=batch_size, epochs=epochs, lr=lr)
        log("Supervised training completed.")

    # ---------------------------
    # Contrastive training
    # ---------------------------
    def train_contrastive(self, texts: List[str], batch_size=16, epochs=5, lr=3e-4, temperature=0.5):
        if not self.model or not self.trainer:
            raise ValueError("Model not initialized. Call init_model() first.")
        self.trainer.train_contrastive(texts, batch_size=batch_size, epochs=epochs, lr=lr, temperature=temperature)
        log("Contrastive training completed.")

    # ---------------------------
    # LoRA fine-tuning
    # ---------------------------
    def init_lora_tuner(self, r=8, alpha=16, dropout=0.1, num_labels: int = 2):
        if not self.model:
            raise ValueError("Model not initialized. Call init_model() first.")
        self.lora_tuner = LoRATuner(base_model_name=self.model.base_model_name, num_labels=num_labels, r=r, alpha=alpha, dropout=dropout, device=self.device)
        log("Initialized LoRA tuner.")

    def train_lora(self, texts: List[str], labels: List[int], batch_size=16, epochs=3, lr=5e-5, save_path="models/lora_model.pt"):
        if not self.lora_tuner:
            raise ValueError("LoRA tuner not initialized. Call init_lora_tuner() first.")
        self.lora_tuner.train(texts, labels, batch_size=batch_size, epochs=epochs, lr=lr, save_path=save_path)
        log(f"LoRA fine-tuning completed. Model saved to {save_path}")

    # ---------------------------
    # Inference
    # ---------------------------
    def encode_texts(self, texts: List[str], batch_size: int = 32):
        if not self.model:
            raise ValueError("Model not initialized. Call init_model() first.")
        return batch_encode_texts(self.model, texts, tokenizer_name=self.model.base_model_name, batch_size=batch_size, device=self.device)

    def predict(self, texts: List[str]):
        if not self.model:
            raise ValueError("Model not initialized. Call init_model() first.")
        self.model.eval()
        predictions = []
        tokenizer = self.model.base.tokenizer if hasattr(self.model.base, 'tokenizer') else None
        if not tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.base_model_name)

        with torch.no_grad():
            for text in texts:
                encoded = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                if self.model.classifier:
                    logits = self.model(input_ids, attention_mask, labels=None)
                    preds = torch.argmax(logits, dim=-1)
                    predictions.append(preds.item())
                else:
                    emb = self.model(input_ids, attention_mask)
                    predictions.append(emb.cpu().numpy())
        return predictions


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    texts = ["Agent executed task A", "Telemetry anomaly detected", "Normal telemetry received"]
    labels = [0, 1, 0]

    pipeline = SentenialMLPipeline()
    pipeline.init_model(embedding_dim=64, num_classes=2)

    # Supervised training
    pipeline.train_supervised(texts, labels, epochs=1)

    # Contrastive training
    pipeline.train_contrastive(texts, epochs=1)

    # LoRA fine-tuning
    pipeline.init_lora_tuner()
    pipeline.train_lora(texts, labels, epochs=1)

    # Inference
    preds = pipeline.predict(["New anomaly detected"])
    print("Predictions:", preds)

    embeddings = pipeline.encode_texts(["New telemetry log"])
    print("Embeddings shape:", embeddings[0].shape if embeddings else None)
