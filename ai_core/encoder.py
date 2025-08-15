# sentenial-x/ai_core/encoder.py
from typing import List
import numpy as np
from .config import EMBEDDING_DIM

class ThreatTextEncoder:
    """
    Lightweight deterministic encoder for logs and telemetry.
    Placeholder for real embeddings (can replace with Sentence-BERT).
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        out = []
        for t in texts:
            rng = abs(hash(t)) % (10**6)
            vec = np.random.default_rng(rng).normal(0, 1, self.dim).astype("float32")
            out.append(vec)
        return np.stack(out, axis=0)
