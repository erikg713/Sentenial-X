# core/semantic_analyzer/models/embedding.py

import numpy as np
from typing import List, Union, Optional
from transformers import AutoTokenizer, AutoModel
import torch
import onnxruntime as ort

from core.semantic_analyzer.models.utils import normalize_vector, batchify
from core.semantic_analyzer.models.registry import ModelRegistry


class EmbeddingModel:
    """
    Real embedding model for semantic analysis.
    Supports:
    - Hugging Face Transformers
    - ONNX Runtime models
    - Batch processing and normalization
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        backend: str = "transformer",  # "transformer" or "onnx"
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.backend = backend
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        self.batch_size = batch_size

        self.model = None
        self.tokenizer = None
        self.session = None

        # Register model
        ModelRegistry.register(self.model_name, lambda: self)

        # Initialize model
        self.load_model()

    def load_model(self, onnx_path: Optional[str] = None):
        """
        Load the real model.
        - If backend == 'transformer', load HuggingFace model.
        - If backend == 'onnx', load ONNX Runtime session.
        """
        if self.backend == "transformer":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
        elif self.backend == "onnx":
            if onnx_path is None:
                raise ValueError("onnx_path must be provided for ONNX backend")
            self.session = ort.InferenceSession(onnx_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        """
        embeddings_list = []

        if self.backend == "transformer":
            for batch in batchify(texts, self.batch_size):
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**encoded)
                    cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                if self.normalize:
                    cls_embeddings = np.array([normalize_vector(vec) for vec in cls_embeddings])

                embeddings_list.append(cls_embeddings)

        elif self.backend == "onnx":
            for batch in batchify(texts, self.batch_size):
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="np")
                ort_inputs = {k: v.astype(np.int64) for k, v in inputs.items()}
                ort_outs = self.session.run(None, ort_inputs)
                batch_embeddings = ort_outs[0]  # Assume first output is embeddings
                if self.normalize:
                    batch_embeddings = np.array([normalize_vector(vec) for vec in batch_embeddings])
                embeddings_list.append(batch_embeddings)

        return np.vstack(embeddings_list)

    def predict(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.encode(texts)


# Example usage
if __name__ == "__main__":
    em = EmbeddingModel(model_name="distilbert-base-uncased", backend="transformer")
    sample_texts = ["Sentenial-X is running", "Real embeddings now"]
    vectors = em.predict(sample_texts)
    print("Embeddings shape:", vectors.shape)
