import logging
from typing import List

try:
    from sentence_transformers import SentenceTransformer
    _has_sentence_transformers = True
except ImportError:
    _has_sentence_transformers = False

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _has_transformers = True
except ImportError:
    _has_transformers = False

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
_DEFAULT_SENTENCE_TRANSFORMER_MODEL = 'all-MiniLM-L6-v2'
_FALLBACK_DIMENSIONS = {
    'sentence_transformers': 384,
    'transformers': 768
}

# Model cache
_model_cache = {}

def load_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def get_embedding(text: str, method: str = 'sentence_transformers', model_name: str = _DEFAULT_SENTENCE_TRANSFORMER_MODEL) -> List[float]:
    """
    Generate an embedding for the input text using the specified method.

    Args:
        text (str): The input text to embed.
        method (str): The embedding method ('sentence_transformers' or 'transformers').
        model_name (str): The model name to use for embedding.

    Returns:
        List[float]: The embedding vector.
    """
    if method == 'sentence_transformers':
        if _has_sentence_transformers:
            try:
                model = load_sentence_transformer_model(model_name)
                embedding = model.encode([text])[0]
                return embedding.tolist()
            except Exception as e:
                logger.error(f"Error generating Sentence Transformer embedding: {e}")
                return [0.0] * _FALLBACK_DIMENSIONS['sentence_transformers']
        else:
            logger.warning("Sentence Transformers library is missing.")
            return [len(text)]  # Fallback

    elif method == 'transformers':
        if _has_transformers:
            try:
                tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                model = AutoModel.from_pretrained("distilbert-base-uncased")
                tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():
                    output = model(**tokens)
                embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
                return embedding
            except Exception as e:
                logger.error(f"Error generating Transformers embedding: {e}")
                return [0.0] * _FALLBACK_DIMENSIONS['transformers']
        else:
            logger.warning("Transformers library is missing.")
            return [len(text)]  # Fallback

    else:
        logger.error(f"Unknown embedding method '{method}'. Using basic fallback.")
        return [len(text)]

if __name__ == '__main__':
    sample_text = "This is a potentially malicious script."

    # Using Sentence Transformers
    embedding_st = get_embedding(sample_text, method='sentence_transformers')
    print(f"\nSentence Transformer embedding for: '{sample_text}'")
    print(f"Length: {len(embedding_st)}")
    print(f"First 10 elements: {embedding_st[:10]}")

    # Using Transformers (DistilBERT)
    embedding_tr = get_embedding(sample_text, method='transformers')
    print(f"\nTransformers (DistilBERT) embedding for: '{sample_text}'")
    print(f"Length: {len(embedding_tr)}")
    print(f"First 10 elements: {embedding_tr[:10]}")

    # Example of fallback
    embedding_fallback = get_embedding(sample_text, method='unknown')
    print(f"\nFallback embedding: {embedding_fallback}")
