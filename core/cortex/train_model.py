"""
Robust, production-friendly training script for fine-tuning BERT on a
cyber-intent classification dataset.

Improvements over the original:
- CLI interface for configurable training (paths, hyperparameters).
- Label mapping supports string labels and persists mapping.
- Mixed precision training (torch.cuda.amp) for faster training and lower memory.
- Learning rate scheduler with warmup.
- Gradient clipping and optional weight decay.
- Evaluation loop with accuracy and validation loss; saves best model.
- Deterministic seeding and sane torch/cudnn settings.
- Logging (python logging) instead of print statements.
- Basic error handling and artifact saving (model, tokenizer, label_map, metrics).
- Type annotations and concise, clear structure for maintainability.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizerFast,
    get_linear_schedule_with_warmup,
)

# ---- Defaults ----
DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_SAVE_DIR = Path("./saved_models/cyber_bert")
DEFAULT_DATA_CSV = Path("data/cyber_intents.csv")
DEFAULT_MAX_LEN = 128
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("train_model")


@dataclass
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    data_csv: Path = DEFAULT_DATA_CSV
    save_dir: Path = DEFAULT_SAVE_DIR
    max_len: int = DEFAULT_MAX_LEN
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    seed: int = DEFAULT_SEED
    num_workers: int = 4
    gradient_clip: float = 1.0
    val_size: float = 0.1
    pin_memory: bool = True


class CyberDataset(Dataset):
    """Simple Dataset wrapper that tokenizes on the fly."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer: BertTokenizerFast, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Use tokenizer that returns tensors then squeeze to keep batch dimension only at collate
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For potential performance improvements; deterministic can hurt perf for some workloads
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def prepare_data(df: pd.DataFrame, val_size: float) -> Tuple[List[str], List[str], List[int], List[int], Dict]:
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    texts = df["text"].astype(str).tolist()
    labels_raw = df["label"].tolist()

    # Map string labels to integers if necessary
    unique_labels = sorted(list({str(l) for l in labels_raw}))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_labels = [label_map[str(l)] for l in labels_raw]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, numeric_labels, test_size=val_size, random_state=0, stratify=numeric_labels if len(unique_labels) > 1 else None
    )
    return train_texts, val_texts, train_labels, val_labels, label_map


def evaluate(model: BertForSequenceClassification, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss.item()
            total_loss += loss * batch["input_ids"].size(0)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["input_ids"].size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(cfg: TrainingConfig) -> None:
    logger.info("Training config: %s", asdict(cfg))
    set_seed(cfg.seed)

    if not cfg.data_csv.exists():
        raise FileNotFoundError(f"Data file not found: {cfg.data_csv}")

    df = pd.read_csv(cfg.data_csv)
    train_texts, val_texts, train_labels, val_labels, label_map = prepare_data(df, val_size=cfg.val_size)
    num_labels = len(label_map)
    logger.info("Found %d classes", num_labels)

    tokenizer = BertTokenizerFast.from_pretrained(cfg.model_name)
    model = BertForSequenceClassification.from_pretrained(cfg.model_name, num_labels=num_labels)
    model.to(DEVICE)

    train_ds = CyberDataset(train_texts, train_labels, tokenizer, cfg.max_len)
    val_ds = CyberDataset(val_texts, val_labels, tokenizer, cfg.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=max(1, cfg.num_workers // 2),
        pin_memory=cfg.pin_memory and torch.cuda.is_available(),
    )

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": cfg.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cfg.lr)

    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    metrics = []

    logger.info("Starting training on device %s", DEVICE)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                loss = outputs.loss
                loss.backward()
                clip_grad_norm_(model.parameters(), cfg.gradient_clip)
                optimizer.step()
            scheduler.step()
            batch_loss = loss.item()
            epoch_loss += batch_loss * batch["input_ids"].size(0)
            pbar.set_postfix(loss=batch_loss)

        avg_train_loss = epoch_loss / len(train_ds)
        val_loss, val_acc = evaluate(model, val_loader, DEVICE)
        metrics.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": val_loss, "val_acc": val_acc})
        logger.info("Epoch %d: train_loss=%.4f val_loss=%.4f val_acc=%.4f", epoch, avg_train_loss, val_loss, val_acc)

        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(cfg.save_dir)
            tokenizer.save_pretrained(cfg.save_dir)
            # persist label map
            with open(cfg.save_dir / "label_map.json", "w", encoding="utf-8") as f:
                json.dump(label_map, f, indent=2, ensure_ascii=False)
            logger.info("Saved best model to %s (val_loss=%.4f)", cfg.save_dir, val_loss)

        # persist metrics each epoch
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(cfg.save_dir / "training_metrics.csv", index=False)

    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a BERT classifier for cyber intents")
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV, help="Path to a CSV file with 'text' and 'label' columns")
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Directory to save model and tokenizer")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Pretrained model name or path")
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--val-size", type=float, default=0.1)
    parsed = parser.parse_args()
    return TrainingConfig(
        model_name=parsed.model_name,
        data_csv=parsed.data_csv,
        save_dir=parsed.save_dir,
        max_len=parsed.max_len,
        batch_size=parsed.batch_size,
        epochs=parsed.epochs,
        lr=parsed.lr,
        weight_decay=parsed.weight_decay,
        warmup_ratio=parsed.warmup_ratio,
        seed=parsed.seed,
        num_workers=parsed.num_workers,
        gradient_clip=parsed.gradient_clip,
        val_size=parsed.val_size,
    )


if __name__ == "__main__":
    cfg = parse_args()
    try:
        train(cfg)
    except Exception as exc:  # pragma: no cover - top level safety
        logger.exception("Training failed: %s", exc)
        raise
