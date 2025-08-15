# scripts/run_pipeline_prod.py
import os
import torch
import logging
from datetime import datetime
from libs.ml.ml_pipeline import MLOrchestrator
from models.lora.lora_tuner import LoRAConfig
from models.distill.distill_trainer import DistillConfig

# ---------------- Logging Setup ----------------
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("SentenialX")

# ---------------- Configs ----------------
BASE_MODEL = "gpt2"               # Base model for LoRA
TEACHER_MODEL = "gpt2-large"     # Teacher for distillation
STUDENT_MODEL = "gpt2"           # Student model
TRAFFIC_MODEL = "bert-base-uncased"

OUTPUT_DIR = "./output_prod"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Placeholder datasets (replace with real CVE, log feeds, or HTTP traffic)
LOCA_TRAIN_TEXTS = [
    "User login failed from IP 192.168.1.10",
    "Suspicious file download detected in /tmp/malware.exe",
    "HTTP POST anomaly detected with unusual payload",
] * 100  # scaled for demonstration

TRAFFIC_SEQUENCES = [
    "GET /login HTTP/1.1 Host: example.com User-Agent: Mozilla/5.0",
    "POST /api/upload HTTP/1.1 Host: example.com Content-Length: 1024",
    "GET /admin HTTP/1.1 Host: example.com Cookie: sessionid=abcd1234",
] * 50

# Hyperparameters
LORA_EPOCHS = 3
DISTILL_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 5e-5

# ---------------- Initialize pipeline ----------------
logger.info("Initializing ML pipeline...")
pipeline = MLOrchestrator()

# ---------------- LoRA Fine-Tuning ----------------
try:
    logger.info("Starting LoRA fine-tuning...")
    lora_config = LoRAConfig(r=8, alpha=16, dropout=0.1)
    pipeline.init_lora(BASE_MODEL, lora_config)
    pipeline.train_lora(
        LOCA_TRAIN_TEXTS,
        output_dir=os.path.join(OUTPUT_DIR, "lora"),
        epochs=LORA_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
except Exception as e:
    logger.exception(f"LoRA fine-tuning failed: {e}")

# ---------------- Distillation ----------------
try:
    logger.info("Starting model distillation...")
    pipeline.init_distill(teacher_model=TEACHER_MODEL, student_model=STUDENT_MODEL)
    distill_config = DistillConfig(temperature=2.0, alpha_ce=0.5, alpha_mse=0.5)
    pipeline.distill(
        LOCA_TRAIN_TEXTS,
        output_dir=os.path.join(OUTPUT_DIR, "distill"),
        config=distill_config,
        epochs=DISTILL_EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE
    )
except Exception as e:
    logger.exception(f"Distillation failed: {e}")

# ---------------- Traffic Encoding ----------------
try:
    logger.info("Starting traffic encoding...")
    pipeline.init_encoder(TRAFFIC_MODEL)
    embeddings = pipeline.encode_traffic(TRAFFIC_SEQUENCES, batch_size=BATCH_SIZE)
except Exception as e:
    logger.exception(f"Traffic encoding failed: {e}")
    embeddings = None

# ---------------- FAISS Indexing ----------------
try:
    if embeddings is not None:
        embedding_dim = embeddings.shape[1]
        pipeline.init_faiss_index(embedding_dim=embedding_dim)
        pipeline.add_to_index(embeddings)
        # Save FAISS index for persistence
        faiss_index_path = os.path.join(OUTPUT_DIR, "faiss_index.bin")
        import faiss
        faiss.write_index(pipeline.vector_index, faiss_index_path)
        logger.info(f"FAISS index saved to {faiss_index_path}")
except Exception as e:
    logger.exception(f"FAISS indexing failed: {e}")

# ---------------- Example Query ----------------
try:
    query_sequences = [
        "POST /api/login HTTP/1.1 Host: example.com User-Agent: test-agent"
    ]
    query_embeddings = pipeline.encode_traffic(query_sequences, batch_size=1)
    distances, indices = pipeline.search_index(query_embeddings, top_k=5)
    logger.info("Query results (top 5 similar sequences):")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        logger.info(f"{i+1}. Sequence index: {idx}, Distance: {dist:.4f}")
except Exception as e:
    logger.exception(f"Query execution failed: {e}")

logger.info("Sentenial-X ML pipeline execution completed successfully.")
