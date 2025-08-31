# core/semantic_analyzer/models/similarity.py

import numpy as np
from typing import Literal


class SimilarityCalculator:
    """
    Computes similarity scores between two embedding vectors.
    Supports multiple metrics: cosine, euclidean, dot product, hybrid.
    """

    def __init__(self, metric: Literal["cosine", "euclidean", "dot", "hybrid"] = "cosine"):
        self.metric = metric

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Converts Euclidean distance to similarity (inverse distance)."""
        distance = np.linalg.norm(a - b)
        return float(1 / (1 + distance))  # Scaled to [0,1]

    def _dot_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Raw dot product similarity."""
        return float(np.dot(a, b))

    def _hybrid_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Weighted combination of cosine and euclidean similarity."""
        cosine = self._cosine_similarity(a, b)
        euclidean = self._euclidean_similarity(a, b)
        return float(0.7 * cosine + 0.3 * euclidean)

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute similarity score between two embeddings using the chosen metric.
        """
        if self.metric == "cosine":
            return self._cosine_similarity(a, b)
        elif self.metric == "euclidean":
            return self._euclidean_similarity(a, b)
        elif self.metric == "dot":
            return self._dot_similarity(a, b)
        elif self.metric == "hybrid":
            return self._hybrid_similarity(a, b)
        else:
            raise ValueError(f"Unknown similarity metric: {self.metric}")


if __name__ == "__main__":
    # Example usage
    vec1 = np.array([0.1, 0.3, 0.5])
    vec2 = np.array([0.1, 0.25, 0.55])

    for metric in ["cosine", "euclidean", "dot", "hybrid"]:
        sim_calc = SimilarityCalculator(metric=metric)
        score = sim_calc.compute(vec1, vec2)
        print(f"{metric.capitalize()} similarity: {score:.4f}")
