# ai_core/embeddings_service.py
from typing import List
from deep_infra_sdk import EmbeddingService
from .utils import preprocess_input

# Initialize embedding model
embedding_model = EmbeddingService(
    model_name="Llama-4-Maverick-17B-128E-Instruct-FP8",
    device_map="auto"
)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for vector search / similarity.
    """
    embeddings = []
    for t in texts:
        embeddings.append(embedding_model.embed(preprocess_input(t)))
    return embeddings
