"""
BERT Embedding Model
Generates sentence or token embeddings for semantic analysis.
"""

from transformers import AutoTokenizer, AutoModel
import torch

class BertEmbedder:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed(self, text: str):
        """Return embedding vector for input text."""
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Use [CLS] token representation as embedding
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
