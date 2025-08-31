# core/semantic_analyzer/models/utils.py

import numpy as np
import torch
from typing import List, Tuple, Union, Dict, Any


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    """
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    v1 = normalize_vector(vec1)
    v2 = normalize_vector(vec2)
    return float(np.dot(v1, v2))


def batchify(data: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split data into smaller batches of size `batch_size`.
    """
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def pad_sequences(sequences: List[List[int]], pad_token: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pads sequences to the same length with `pad_token`.

    Returns:
        padded: np.ndarray of shape (batch_size, max_length)
        mask: np.ndarray of shape (batch_size, max_length), where 1=real token, 0=padding
    """
    max_len = max(len(seq) for seq in sequences)
    padded = np.full((len(sequences), max_len), pad_token, dtype=np.int64)
    mask = np.zeros((len(sequences), max_len), dtype=np.float32)

    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
        mask[i, :len(seq)] = 1.0

    return padded, mask


def to_tensor(data: Union[np.ndarray, List], device: str = "cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Convert numpy array or list into a Torch tensor.
    """
    return torch.tensor(data, dtype=dtype, device=device)


def from_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert Torch tensor to numpy array (detach and move to CPU if necessary).
    """
    return tensor.detach().cpu().numpy()


def prepare_onnx_inputs(inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Ensure ONNX Runtime inputs are numpy arrays.
    """
    return {k: (v if isinstance(v, np.ndarray) else np.array(v)) for k, v in inputs.items()}


def top_k_indices(scores: np.ndarray, k: int = 5) -> List[int]:
    """
    Return indices of the top-k highest values in a numpy array.
    """
    return scores.argsort()[-k:][::-1].tolist()


def mean_pooling(embeddings: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Mean pooling of embeddings, ignoring padding tokens.

    Args:
        embeddings: np.ndarray of shape (batch_size, seq_len, hidden_dim)
        mask: np.ndarray of shape (batch_size, seq_len)

    Returns:
        np.ndarray of shape (batch_size, hidden_dim)
    """
    mask_expanded = np.expand_dims(mask, -1)
    summed = np.sum(embeddings * mask_expanded, axis=1)
    counts = np.sum(mask_expanded, axis=1)
    counts[counts == 0] = 1
    return summed / counts
