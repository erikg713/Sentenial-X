# scripts/run_pipeline.py

from libs.ml.ml_pipeline import SentenialMLPipeline

# ---------------------------
# Sample telemetry/log data
# ---------------------------
texts = [
    "Agent executed task A",
    "Telemetry anomaly detected",
    "Normal telemetry received",
    "VSSAdmin delete shadows detected",
    "Process injected with AMSI bypass"
]
labels = [0, 1, 0, 1, 1]  # 0 = benign, 1 = malicious

# ---------------------------
# Initialize ML pipeline
# ---------------------------
pipeline = SentenialMLPipeline()
pipeline.init_model(base_model_name="bert-base-uncased", embedding_dim=64, num_classes=2)

# ---------------------------
# Supervised training
# ---------------------------
print("\n--- Supervised Training ---")
pipeline.train_supervised(texts, labels, batch_size=2, epochs=2, lr=5e-5)

# ---------------------------
# Contrastive training (self-supervised)
# ---------------------------
print("\n--- Contrastive Training ---")
pipeline.train_contrastive(texts, batch_size=2, epochs=2, lr=3e-4, temperature=0.5)

# ---------------------------
# LoRA fine-tuning
# ---------------------------
print("\n--- LoRA Fine-Tuning ---")
pipeline.init_lora_tuner(r=4, alpha=8, dropout=0.1, num_labels=2)
pipeline.train_lora(texts, labels, batch_size=2, epochs=2, lr=5e-5, save_path="models/lora_telemetry.pt")

# ---------------------------
# Inference examples
# ---------------------------
print("\n--- Inference ---")
new_texts = [
    "Suspicious AMSI bypass detected",
    "Routine telemetry event"
]

predictions = pipeline.predict(new_texts)
print("Predictions (0=benign, 1=malicious):", predictions)

embeddings = pipeline.encode_texts(new_texts)
print("Embeddings shape:", embeddings.shape)
