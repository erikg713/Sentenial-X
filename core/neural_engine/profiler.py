# core/neural_engine/profiler.py
try:
    from sentence_transformers import SentenceTransformer
    _has_sentence_transformers = True
except ImportError:
    _has_sentence_transformers = False
    print("Warning: Sentence Transformers library not found.")

try:
    from transformers import AutoModel, AutoTokenizer
    import torch
    _has_transformers = True
    _distilbert_tokenizer = None
    _distilbert_model = None
except ImportError:
    _has_transformers = False
    print("Warning: Transformers library not found.")

_default_sentence_transformer_model = 'all-MiniLM-L6-v2'

def get_embedding(text, method='sentence_transformers', sentence_transformer_model=_default_sentence_transformer_model):
    if method == 'sentence_transformers':
        if _has_sentence_transformers:
            global _loaded_model
            try:
                if _loaded_model is None or _loaded_model.name != sentence_transformer_model:
                    _loaded_model = SentenceTransformer(sentence_transformer_model)
                embedding = _loaded_model.encode([text])[0]
                return embedding.tolist()
            except Exception as e:
                print(f"Error generating Sentence Transformer embedding: {e}")
                return [0.0] * 384  # Default size for 'all-MiniLM-L6-v2'
        else:
            print("Warning: Sentence Transformers library is missing.")
            return [len(text)]  # Fallback

    elif method == 'transformers':
        if _has_transformers:
            global _distilbert_tokenizer, _distilbert_model
            try:
                if _distilbert_tokenizer is None:
                    _distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
                if _distilbert_model is None:
                    _distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")

                tokens = _distilbert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():
                    output = _distilbert_model(**tokens)
                embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
                return embedding
            except Exception as e:
                print(f"Error generating Transformers embedding: {e}")
                return [0.0] * 768  # Default size for distilbert-base-uncased
        else:
            print("Warning: Transformers library is missing.")
            return [len(text)]  # Fallback

    else:
        print(f"Error: Unknown embedding method '{method}'. Using basic fallback.")
        return [len(text)]

if __name__ == '__main__':
    sample_text = "This is a potentially malicious script."

    # Using Sentence Transformers
    embedding_st = get_embedding(sample_text, method='sentence_transformers')
    print(f"\nSentence Transformer embedding for: '{sample_text}'")
    if _has_sentence_transformers:
        print(f"Length: {len(embedding_st)}")
        print(f"First 10 elements: {embedding_st[:10]}")
    else:
        print(f"Basic embedding: {embedding_st}")

    # Using Transformers (DistilBERT)
    embedding_tr = get_embedding(sample_text, method='transformers')
    print(f"\nTransformers (DistilBERT) embedding for: '{sample_text}'")
    if _has_transformers:
        print(f"Length: {len(embedding_tr)}")
        print(f"First 10 elements: {embedding_tr[:10]}")
    else:
        print(f"Basic embedding: {embedding_tr}")

    # Example of fallback
    embedding_fallback = get_embedding(sample_text, method='unknown')
    print(f"\nFallback embedding: {embedding_fallback}")
class EmbeddingModelManager:
    def __init__(self):
        self._sentence_transformer_model = None
        self._distilbert_tokenizer = None
        self._distilbert_model = None
    
    def get_sentence_transformer_embedding(self, text: str, model_name: str) -> list:
        try:
            if self._sentence_transformer_model is None or self._sentence_transformer_model.name != model_name:
                self._sentence_transformer_model = SentenceTransformer(model_name)
            embedding = self._sentence_transformer_model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating Sentence Transformer embedding: {e}")
            return [0.0] * 384

    def get_transformers_embedding(self, text: str) -> list:
        try:
            if self._distilbert_tokenizer is None:
                self._distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            if self._distilbert_model is None:
                self._distilbert_model = AutoModel.from_pretrained("distilbert-base-uncased")
            
            tokens = self._distilbert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            with torch.no_grad():
                output = self._distilbert_model(**tokens)
            embedding = output.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error generating Transformers embedding: {e}")
            return [0.0] * 768
