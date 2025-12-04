# sentenial_x/core/cortex/model_loader.py
"""
Improved model loader and inference helpers for the cyber intent classifier.

Features:
- Safe device selection with CUDA fallback
- Lazy or eager model/tokenizer loading
- Batch and single-text prediction APIs
- Probability outputs and optional label mapping
- Resource cleanup helper
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union, Dict

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from .config import CONFIG

logger = logging.getLogger(__name__)


class CyberIntentModel:
    """
    Wrapper around a Hugging Face sequence classification model providing
    safe loading and convenient inference helpers.

    Usage:
      cfg = CONFIG["model"]
      m = CyberIntentModel(load_on_init=True)
      label = m.predict("some text")
      probs = m.predict_proba("some text")
      batch = m.predict_batch(["one", "two"])
      m.close()  # free GPU memory if needed

    The class will use CONFIG["model"]["label_map"] (dict) if present to map
    predicted indices to human-readable labels.
    """

    def __init__(self, load_on_init: bool = True) -> None:
        model_cfg = CONFIG.get("model", {})
        requested_device = model_cfg.get("device", "cpu")
        self.device = self._select_device(requested_device)
        self.model_path: str = model_cfg.get("custom_model_path", "bert-base-uncased")
        self.max_len: int = int(model_cfg.get("max_seq_length", 128))
        self.label_map: Optional[Dict[int, str]] = model_cfg.get("label_map") or None

        self._tokenizer = None
        self._model = None

        if load_on_init:
            self._load()

    # ---- internal helpers ----
    def _select_device(self, requested: str) -> torch.device:
        requested_lower = requested.lower()
        if "cuda" in requested_lower or "gpu" in requested_lower:
            if torch.cuda.is_available():
                logger.debug("Using CUDA device for model inference.")
                return torch.device("cuda")
            logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")

    def _load(self) -> None:
        """Load tokenizer and model into memory and move model to device."""
        if self._tokenizer is not None and self._model is not None:
            return  # already loaded

        try:
            logger.info("Loading tokenizer from %s", self.model_path)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

            logger.info("Loading model from %s", self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self.device)
            self._model.eval()
            logger.info("Model loaded and moved to %s", self.device)
        except Exception as exc:
            logger.exception("Failed to load model/tokenizer from %s", self.model_path)
            raise RuntimeError(f"Failed to load model/tokenizer: {exc}") from exc

    # ---- public API ----
    def ensure_loaded(self) -> None:
        """Ensure the model and tokenizer are loaded (lazy load helper)."""
        if self._model is None or self._tokenizer is None:
            self._load()

    def predict(
        self, text: str, return_label: bool = True
    ) -> Union[str, int]:
        """
        Predict the class for a single text.

        Returns:
          - a label string if return_label is True and label_map is available,
          - otherwise returns the predicted class index (int).
        """
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")

        preds = self.predict_batch([text], return_label=return_label)
        return preds[0]

    def predict_batch(
        self, texts: List[str], return_label: bool = True, batch_size: int = 32
    ) -> List[Union[str, int]]:
        """
        Predict classes for a list of texts.

        Args:
          texts: list of input strings (must not be empty)
          return_label: whether to map indices to labels (requires CONFIG label_map)
          batch_size: inference batch size (controls memory/speed tradeoff)

        Returns:
          List of predictions (labels or indices).
        """
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("texts must be a non-empty list of strings")

        self.ensure_loaded()

        results: List[Union[str, int]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                max_length=self.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().tolist()

            if return_label and self.label_map:
                results.extend([self.label_map.get(p, str(p)) for p in preds])
            elif return_label and not self.label_map:
                # Return string for backward compatibility with original code.
                results.extend([str(p) for p in preds])
            else:
                results.extend(preds)

        return results

    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Return a mapping of label (or index) to probability for a single text.

        If a label_map exists, keys will be label strings; otherwise keys are
        stringified class indices.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")

        self.ensure_loaded()

        inputs = self._tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits.squeeze(0)  # shape: (num_labels,)
            probs = F.softmax(logits, dim=-1).cpu().tolist()

        # map to labels if available
        mapped: Dict[str, float] = {}
        for idx, p in enumerate(probs):
            key = self.label_map.get(idx, str(idx)) if self.label_map else str(idx)
            mapped[key] = float(p)

        return mapped

    def close(self) -> None:
        """
        Free model resources. After calling this, object can be re-used but will
        lazy-load the model again on next inference.
        """
        if self._model is not None:
            try:
                # Move to CPU before deleting to avoid CUDA memory quirks.
                try:
                    self._model.to("cpu")
                except Exception:
                    pass
                del self._model
                self._model = None
            except Exception:
                logger.exception("Error while deleting the model object")
        if self._tokenizer is not None:
            try:
                del self._tokenizer
                self._tokenizer = None
            except Exception:
                logger.exception("Error while deleting the tokenizer object")

        # Clear CUDA cache if applicable
        if self.device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.exception("Failed to empty CUDA cache")

    # Alias for backward compatibility
    def predict_raw(self, text: str) -> str:
        """
        Backward-compatible wrapper that mirrors original behavior:
        returns the predicted class index as a string.
        """
        pred = self.predict(text, return_label=False)
        return str(pred)
