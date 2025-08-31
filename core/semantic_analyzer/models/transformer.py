# core/semantic_analyzer/models/transformer.py

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Any, Optional


class TransformerModel(nn.Module):
    """
    Transformer-based model wrapper for semantic analysis.
    Provides an abstraction around Hugging Face transformer models
    for embeddings, classification, and sequence encoding.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: Optional[str] = None,
        output_hidden_states: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_hidden_states = output_hidden_states

        # Load tokenizer and transformer backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, output_hidden_states=output_hidden_states)
        self.transformer.to(self.device)

        # Optional classification head (lazy init)
        self.classifier_head: Optional[nn.Module] = None

    def add_classification_head(self, num_labels: int):
        """
        Dynamically attach a classification head for fine-tuning tasks.
        """
        hidden_size = self.transformer.config.hidden_size
        self.classifier_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        ).to(self.device)

    def forward(
        self,
        texts: List[str],
        return_embeddings: bool = True,
        return_logits: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the transformer model on input text.

        Args:
            texts: List of strings to process.
            return_embeddings: Whether to return embeddings.
            return_logits: Whether to return classification logits (if classifier head exists).

        Returns:
            Dictionary containing embeddings, logits, and/or hidden states.
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(self.device)

        outputs = self.transformer(**encoded)

        # CLS token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        result: Dict[str, Any] = {}

        if return_embeddings:
            result["embeddings"] = cls_embeddings.detach().cpu()

        if self.classifier_head and return_logits:
            logits = self.classifier_head(cls_embeddings)
            result["logits"] = logits.detach().cpu()

        if self.output_hidden_states:
            result["hidden_states"] = [h.detach().cpu() for h in outputs.hidden_states]

        return result

    def encode(self, texts: List[str], normalize: bool = True) -> torch.Tensor:
        """
        Return embeddings for given texts.
        """
        result = self.forward(texts, return_embeddings=True)
        embeddings = result["embeddings"]

        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def classify(self, texts: List[str]) -> torch.Tensor:
        """
        Return logits for classification (requires classifier head).
        """
        if not self.classifier_head:
            raise RuntimeError("Classification head not initialized. Call add_classification_head().")

        result = self.forward(texts, return_embeddings=False, return_logits=True)
        return result["logits"]

    def save(self, path: str):
        """
        Save transformer and optional classifier head.
        """
        self.transformer.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        if self.classifier_head:
            torch.save(self.classifier_head.state_dict(), f"{path}/classifier_head.pt")

    def load(self, path: str):
        """
        Load transformer and optional classifier head.
        """
        self.transformer = AutoModel.from_pretrained(path, output_hidden_states=self.output_hidden_states).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        try:
            state_dict = torch.load(f"{path}/classifier_head.pt", map_location=self.device)
            hidden_size = self.transformer.config.hidden_size
            self.classifier_head = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(hidden_size, len(state_dict["1.weight"]))
            ).to(self.device)
            self.classifier_head.load_state_dict(state_dict)
        except FileNotFoundError:
            self.classifier_head = None
