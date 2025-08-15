# sentenial-x/ai_core/utils.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def similarity_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    """
    return float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0])
