"""
Sentenial-X AI Core: Encoder Module
-----------------------------------
Responsible for encoding text, payloads, and logs into vector embeddings
or structured features for AI inference and threat analysis.

Author: Sentenial-X Development Team
"""

from typing import List, Union
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from api.utils.logger import init_logger

logger = init_logger("ai_core.encoder")


class Encoder:
    """
    Provides encoding utilities for text, logs, and events.
    Supports TF-IDF, hash-based embeddings, and numeric feature vectors.
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._fitted = False
        logger.info("Encoder initialized")

    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Convert text or list of texts into TF-IDF vector embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not self._fitted:
            # Fit vectorizer on provided texts
            self._vectorizer.fit(texts)
            self._fitted = True
            logger.info("TF-IDF vectorizer fitted on input texts")

        vectors = self._vectorizer.transform(texts).toarray()
        logger.debug(f"Encoded {len(texts)} text items into vectors of size {vectors.shape[1]}")
        return vectors

    def encode_hash(self, data: Union[str, bytes]) -> str:
        """
        Deterministic hash encoding for logs or events.
        Returns a SHA256 hex digest.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        hashed = hashlib.sha256(data).hexdigest()
        logger.debug(f"Hashed data to {hashed}")
        return hashed

    def encode_numeric_features(self, data: List[float]) -> np.ndarray:
        """
        Convert numeric list into a normalized numpy array.
        """
        arr = np.array(data, dtype=np.float32)
        if len(arr) == 0:
            return arr
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        logger.debug(f"Encoded numeric features: {arr}")
        return arr

    def batch_encode(self, data_list: List[str]) -> np.ndarray:
        """
        Batch encode multiple text inputs into vectors.
        """
        return self.encode_text(data_list)


# ------------------------
# Quick CLI Test
# ------------------------
if __name__ == "__main__":
    encoder = Encoder()
    texts = [
        "Detected SQL injection attempt on /login endpoint",
        "User admin logged in successfully",
        "Suspicious XSS payload <script>alert(1)</script>",
    ]
    vectors = encoder.encode_text(texts)
    for i, vec in enumerate(vectors):
        print(f"Text {i} vector length: {len(vec)}")

    hashed = encoder.encode_hash("example payload")
    print("Hash:", hashed)

    numeric = encoder.encode_numeric_features([1, 2, 3, 4])
    print("Numeric vector:", numeric)
