import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class HTTPEncoder(nn.Module):
    """
    Simple HTTP traffic encoder: serialize JSON payload into
    fixed-size embedding via a pretrained text model.
    """
    def __init__(self, pretrained: str, embed_dim: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.encoder = AutoModel.from_pretrained(pretrained)
        # Optional projection if dimensions mismatch
        self.project = nn.Linear(self.encoder.config.hidden_size, embed_dim)

    def forward(self, http_json: str) -> torch.Tensor:
        """
        http_json: a JSON-string of headers+payload
        returns: (batch_size, embed_dim) Tensor
        """
        inputs = self.tokenizer(
            http_json, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.encoder(**inputs)
        # mean-pool over sequence
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.project(pooled)

