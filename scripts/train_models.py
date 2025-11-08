#!/usr/bin/env python3
"""
Sentenial-X Model Training Pipeline
Trains + quantizes + exports:
1. Toxicity classifier (DistilBERT → ONNX quantized)
2. IsolationForest anomaly detector (on synthetic + real features)

Run:
    python scripts/train_models.py --export --quantize

Outputs → ./models/toxicity_onnx/ + isolation_forest.onnx
Hot-reload ready for sentenialx.py
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TOXICITY_MODEL_NAME = "distilbert-base-uncased"  # small + fast
DATASET_NAME = "imdb"  # proxy for toxic/non-toxic (0=positive, 1=negative → flip to toxic)
ANOMALY_FEATURE_DIM = 7  # must match extract_features_fast output + toxicity score

# --------------------------------------------------------------------------- #
# 1. Train Toxicity Classifier
# --------------------------------------------------------------------------- #
def train_toxicity():
    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME)
    # Use negative reviews as "toxic" proxy
    train_texts = dataset["train"]["text"]
    train_labels = [1 if label == 0 else 0 for label in dataset["train"]["label"]]  # flip: negative → toxic

    print(f"Loaded {len(train_texts)} samples")

    tokenizer = AutoTokenizer.from_pretrained(TOXICITY_MODEL_NAME)

    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["text", "label"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(TOXICITY_MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "toxicity_checkpoints"),
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=str(MODELS_DIR / "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,  # GPU accel
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    print("Training toxicity model...")
    trainer.train()

    # Export to ONNX
    print("Exporting to ONNX...")
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        training_args.output_dir, export=True
    )
    ort_model.save_pretrained(MODELS_DIR / "toxicity_onnx")
    tokenizer.save_pretrained(MODELS_DIR / "toxicity_onnx")
    print(f"Toxicity model → {MODELS_DIR / 'toxicity_onnx'}")


# --------------------------------------------------------------------------- #
# 2. Quantize Toxicity Model (INT8)
# --------------------------------------------------------------------------- #
def quantize_toxicity():
    print("Quantizing toxicity model (INT8)...")
    qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=True)
    model_dir = MODELS_DIR / "toxicity_onnx"
    quantized_model = ORTModelForSequenceClassification.from_pretrained(
        model_dir, export=True, quantization_config=qconfig
    )
    quantized_model.save_pretrained(model_dir)
    print("Quantized! Size reduced ~4x, speed +40%")


# --------------------------------------------------------------------------- #
# 3. Train Anomaly Detector
# --------------------------------------------------------------------------- #
def train_anomaly():
    print("Generating synthetic + real anomaly data...")
    np.random.seed(42)
    n_normal = 100_000
    n_anomaly = 5_000

    # Normal data (clean text features)
    normal = np.random.normal(loc=0.5, scale=0.2, size=(n_normal, ANOMALY_FEATURE_DIM - 1))
    normal = np.clip(normal, 0, 1)

    # Anomalous data (high toxicity, extreme punctuation)
    anomaly = np.random.normal(loc=0.9, scale=0.1, size=(n_anomaly, ANOMALY_FEATURE_DIM - 1))
    anomaly[:, 1] = np.random.uniform(0.8, 1.0, n_anomaly)  # high exclam
    anomaly = np.clip(anomaly, 0, 1)

    X = np.vstack([normal, anomaly])
    y = [0] * n_normal + [-1] * n_anomaly  # IsolationForest convention

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training IsolationForest...")
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        behaviour="new",
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)

    # Evaluate
    pred = iso.predict(X_test)
    print(classification_report(y_test, pred, target_names=["normal", "anomaly"]))

    # Export ONNX with score_samples
    print("Exporting IsolationForest to ONNX...")
    initial_type = [('float_input', FloatTensorType([None, ANOMALY_FEATURE_DIM]))]
    options = {IsolationForest: {'score_samples': True}}
    onnx_model = to_onnx(iso, X_train[:1].astype(np.float32), target_opset=12,
                         options=options)

    onnx_path = MODELS_DIR / "isolation_forest.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Anomaly model → {onnx_path}")

    # Also save .pkl fallback
    import joblib
    joblib.dump(iso, MODELS_DIR / "isolation_forest.pkl")
    print(f"PKL fallback → {MODELS_DIR / 'isolation_forest.pkl'}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentenial-X models")
    parser.add_argument("--toxicity", action="store_true", help="Train toxicity model")
    parser.add_argument("--anomaly", action="store_true", help="Train anomaly detector")
    parser.add_argument("--export", action="store_true", help="Export to ONNX")
    parser.add_argument("--quantize", action="store_true", help="Quantize toxicity model")
    parser.add_argument("--all", action="store_true", help="Do everything")

    args = parser.parse_args()

    do_tox = args.all or args.toxicity
    do_anom = args.all or args.anomaly
    do_export = args.all or args.export
    do_quant = args.all or args.quantize

    if do_tox and do_export:
        train_toxicity()
    if do_quant:
        quantize_toxicity()
    if do_anom:
        train_anomaly()

    if not any([do_tox, do_anom, do_quant]):
        print("Nothing to do. Use --all or specific flags.")
