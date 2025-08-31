"""
core/semantic_analyzer/models/embedding.py

Embedding model interface for Sentenial-X.
Supports OpenAI embeddings, HuggingFace transformers, and deterministic fallback.
Includes caching integration for performance optimization.
"""

import hashlib
import logging
from typing import List, Optional, Dict, Any

from core.semantic_analyzer.models.cache import EmbeddingCache

logger = logging.getLogger(__name__)


class BaseEmbedder:
    """Abstract base class for all embedding models."""

    def embed(self, text: str) -> List[float]:
        raise NotImplementedError("Embed method must be implemented.")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


class HashEmbedder(BaseEmbedder):
    """
    Deterministic, lightweight fallback embedding generator.
    Uses SHA256 hashing to produce fixed-length pseudo-embeddings.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [b / 255.0 for b in digest[: self.dim]]
        # Pad to required dimension if digest is shorter
        while len(vector) < self.dim:
            vector.append(0.0)
        return vector


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI Embedding Model wrapper.
    Requires OPENAI_API_KEY in environment.
    """

    def __init__(self, model: str = "text-embedding-ada-002"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        import os

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(model=self.model, input=text)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return []


class HuggingFaceEmbedder(BaseEmbedder):
    """
    HuggingFace Transformers embedding model.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"HuggingFace embedding failed: {e}")
            return []


class EmbeddingModel:
    """
    Unified embedding interface with caching support.
    Falls back gracefully if preferred backend fails.
    """

    def __init__(
        self,
        backend: str = "huggingface",
        cache: Optional[EmbeddingCache] = None,
        **kwargs: Any,
    ):
        self.cache = cache or EmbeddingCache()
        self.backend_name = backend.lower()

        if self.backend_name == "openai":
            self.model = OpenAIEmbedder(**kwargs)
        elif self.backend_name == "huggingface":
            self.model = HuggingFaceEmbedder(**kwargs)
        else:
            logger.warning("Using fallback HashEmbedder (not recommended for semantic tasks).")
            self.model = HashEmbedder()

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single string with caching."""
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        vector = self.model.embed(text)
        if vector:
            self.cache.set(text, vector)
        return vector

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with caching."""
        results = []
        for t in texts:
            results.append(self.embed(t))
        return results
