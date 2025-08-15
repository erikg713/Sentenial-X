# scripts/run_pipeline.py
import torch
from libs.ml.ml_pipeline import MLOrchestrator
from models.lora.lora_tuner import LoRAConfig
from models.distill.distill_trainer import DistillConfig

# ---------------- Configs ----------------
BASE_MODEL = "gpt2"               # Base model for LoRA
TEACHER_MODEL = "gpt2-large"     # Teacher for distillation
STUDENT_MODEL = "gpt2"           # Student model
TRAFFIC_MODEL = "bert-base-uncased"

LOCA_TRAIN_TEXTS = [
    "Example log entry: user login failed from IP 192.168.1.10",
    "Suspicious file download detected in /tmp/malware.exe",
    "HTTP request anomaly: unusual POST payload pattern"
]

TRAFFIC_SEQUENCES = [
    "GET /login HTTP/1.1 Host: example.com User-Agent: Mozilla/5.0",
    "POST /api/upload HTTP/1.1 Host: example.com Content-Length: 1024",
    "GET /admin HTTP/1.1 Host: example.com Cookie: sessionid=abcd1234"
]

OUTPUT_DIR = "./output"

# ---------------- Initialize orchestrator ----------------
pipeline = MLOrchestrator()

# ---------------- LoRA Fine-Tuning ----------------
lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1)
pipeline.init_lora(BASE_MODEL, lora_config)
pipeline.train_lora(LOCA_TRAIN_TEXTS, output_dir=f"{OUTPUT_DIR}/lora", epochs=1, batch_size=2)

# ---------------- Distillation ----------------
pipeline.init_distill(teacher_model=TEACHER_MODEL, student_model=STUDENT_MODEL)
distill_config = DistillConfig(temperature=2.0, alpha_ce=0.5, alpha_mse=0.5)
pipeline.distill(LOCA_TRAIN_TEXTS, output_dir=f"{OUTPUT_DIR}/distill", config=distill_config, epochs=1, batch_size=2)

# ---------------- Traffic Encoding ----------------
pipeline.init_encoder(TRAFFIC_MODEL)
embeddings = pipeline.encode_traffic(TRAFFIC_SEQUENCES, batch_size=2)

# ---------------- FAISS Index ----------------
pipeline.init_faiss_index(embedding_dim=embeddings.shape[1])
pipeline.add_to_index(embeddings)

# Example query
query_embeddings = pipeline.encode_traffic(["POST /api/login HTTP/1.1 Host: example.com"])
distances, indices = pipeline.search_index(query_embeddings, top_k=2)

print("Query results (top 2 similar sequences):")
for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
    print(f"{i+1}. Sequence index: {idx}, Distance: {dist:.4f}")
