# libs/ml/ml_pipeline.py
import torch
import os
from typing import List, Optional

from models.lora.lora_tuner import LoRATuner, LoRAConfig
from models.distill.distill_trainer import DistillTrainer, DistillConfig
from models.encoder.traffic_encoder import TrafficEncoder

# Optional: vector DB
try:
    import faiss
except ImportError:
    faiss = None

class MLOrchestrator:
    """
    Full ML pipeline orchestrator for Sentenial-X.
    Handles:
        1. LoRA fine-tuning
        2. Model distillation
        3. HTTP/traffic embedding extraction
        4. Optional FAISS vector DB indexing
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lora_tuner: Optional[LoRATuner] = None
        self.distill_trainer: Optional[DistillTrainer] = None
        self.traffic_encoder: Optional[TrafficEncoder] = None
        self.vector_index = None

    # ---------------- LoRA ----------------
    def init_lora(self, base_model: str, lora_config: LoRAConfig):
        self.lora_tuner = LoRATuner(model_name=base_model, lora_config=lora_config, device=self.device)
        print("[MLOrchestrator] LoRA tuner initialized.")

    def train_lora(self, texts: List[str], output_dir: str, epochs: int = 3, batch_size: int = 8, lr: float = 5e-5):
        dataset = self.lora_tuner.tokenize_dataset(texts)
        self.lora_tuner.train(dataset, output_dir, epochs=epochs, batch_size=batch_size, lr=lr)
        print(f"[MLOrchestrator] LoRA training finished. Checkpoint saved to {output_dir}")

    # ---------------- Distillation ----------------
    def init_distill(self, teacher_model: str, student_model: str):
        self.distill_trainer = DistillTrainer(teacher_model_name=teacher_model, student_model_name=student_model, device=self.device)
        print("[MLOrchestrator] Distillation trainer initialized.")

    def distill(self, texts: List[str], output_dir: str, config: Optional[DistillConfig] = None, epochs: int = 3, batch_size: int = 8, lr: float = 5e-5):
        dataset = self.distill_trainer.tokenize_dataset(texts)
        config = config or DistillConfig()
        self.distill_trainer.train(dataset, output_dir, config=config, epochs=epochs, batch_size=batch_size, lr=lr)
        print(f"[MLOrchestrator] Distillation complete. Student model saved to {output_dir}")

    # ---------------- Traffic Encoder ----------------
    def init_encoder(self, model_name: str = "bert-base-uncased"):
        self.traffic_encoder = TrafficEncoder(model_name=model_name, device=self.device)
        print("[MLOrchestrator] Traffic encoder initialized.")

    def encode_traffic(self, sequences: List[str], batch_size: int = 16):
        embeddings = self.traffic_encoder.encode(sequences, batch_size=batch_size)
        print(f"[MLOrchestrator] Encoded {len(sequences)} sequences into embeddings of shape {embeddings.shape}")
        return embeddings

    # ---------------- Vector DB ----------------
    def init_faiss_index(self, embedding_dim: int):
        if faiss is None:
            raise ImportError("faiss not installed. Run `pip install faiss-cpu` or `faiss-gpu`.")
        self.vector_index = faiss.IndexFlatL2(embedding_dim)
        print(f"[MLOrchestrator] FAISS index initialized with dimension {embedding_dim}")

    def add_to_index(self, embeddings: torch.Tensor):
        if self.vector_index is None:
            raise ValueError("Vector index not initialized. Call init_faiss_index first.")
        embeddings_np = embeddings.detach().cpu().numpy()
        self.vector_index.add(embeddings_np)
        print(f"[MLOrchestrator] Added {embeddings_np.shape[0]} embeddings to FAISS index.")

    def search_index(self, query_embeddings: torch.Tensor, top_k: int = 5):
        if self.vector_index is None:
            raise ValueError("Vector index not initialized. Call init_faiss_index first.")
        query_np = query_embeddings.detach().cpu().numpy()
        distances, indices = self.vector_index.search(query_np, top_k)
        return distances, indices
