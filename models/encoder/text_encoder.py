# sentenialx/models/encoder/text_encoder.py
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

class ThreatTextEncoder:
    """
    ThreatTextEncoder: encodes threat logs, system events, or payload text
    into dense embeddings suitable for anomaly detection or FAISS indexing.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded_input = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)

            # Mean pooling
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            batch_embeddings = (summed / counts).cpu().numpy()
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)
