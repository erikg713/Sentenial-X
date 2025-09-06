# ai_core/threat_analyzer.py
from .predictive_model import enqueue_task, select_model
from .embeddings_service import generate_embeddings
from .utils import preprocess_input

async def analyze_threat(text: str, complexity: str = "medium") -> dict:
    processed = preprocess_input(text)
    result = await enqueue_task(processed, complexity)
    embedding = generate_embeddings([processed])[0]
    return {
        "model_used": select_model(complexity).model_name,
        "analysis": result,
        "embedding": embedding
    }
