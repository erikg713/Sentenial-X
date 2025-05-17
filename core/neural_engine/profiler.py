from transformers import AutoModel, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def get_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state.mean(dim=1).squeeze().numpy()
# core/neural_engine/profiler.py
try:
    from sentence_transformers import SentenceTransformer
    _has_sentence_transformers = True
except ImportError:
    _has_sentence_transformers = False
    print("Warning: Sentence Transformers library not found. Embedding functionality will be limited.")

_default_embedding_model = 'all-MiniLM-L6-v2'
_loaded_model = None

def get_embedding(text, model_name=_default_embedding_model):
    global _loaded_model
    if _has_sentence_transformers:
        try:
            if _loaded_model is None or _loaded_model.name != model_name:
                _loaded_model = SentenceTransformer(model_name)
            embedding = _loaded_model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 384  # Return a zero vector of the expected size for the default model
    else:
        print("Warning: Cannot generate embeddings. Sentence Transformers library is missing.")
        # Fallback to a very basic "embedding" if the library is not available
        return [len(text)]

if __name__ == '__main__':
    # Example usage
    sample_text = "This is a potentially malicious script."
    embedding = get_embedding(sample_text)
    print(f"Embedding for: '{sample_text}'")
    if _has_sentence_transformers:
        print(f"Length of embedding: {len(embedding)}")
        print(f"First 10 elements: {embedding[:10]}")
    else:
        print(f"Basic embedding: {embedding}")

    another_text = "A normal system log entry."
    embedding_2 = get_embedding(another_text)
    print(f"\nEmbedding for: '{another_text}'")
    if _has_sentence_transformers:
        print(f"Length of embedding: {len(embedding_2)}")
        print(f"First 10 elements: {embedding_2[:10]}")
    else:
        print(f"Basic embedding: {embedding_2}")
