# sentenialx/models/encoder/text_encoder.py
from typing import List
import numpy as np

class ThreatTextEncoder:
    def __init__(self, vocab_size: int = 50000, dim: int = 256):
        self.vocab_size = vocab_size
        self.dim = dim

    def encode(self, texts: List[str]) -> np.ndarray:
        # TODO: replace with real tokenizer + embedding model
        # For now: deterministic hash -> pseudo-embedding
        out = []
        for t in texts:
            rng = abs(hash(t)) % (10**6)
            vec = np.random.default_rng(rng).normal(0, 1, self.dim).astype("float32")
            out.append(vec)
        return np.stack(out, axis=0)
