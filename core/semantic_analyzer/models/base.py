""" core/semantic_analyzer/models/base.py

Abstract base classes and interfaces for models used by the semantic analyzer.

This file defines a small, dependency-light contract that all model implementations in core.semantic_analyzer.models should follow. It makes it easy to swap implementations (Transformer, ONNX, lightweight fallbacks) while keeping the rest of the pipeline consistent.

Design goals:

Minimal runtime dependencies (typing + abc only)

Clear lifecycle hooks: load, unload, predict/embed, persist

Optional async support where useful

Lightweight metadata & config passing """ from future import annotations from abc import ABC, abstractmethod from typing import Any, Dict, Optional, Sequence


class BaseModel(ABC): """Abstract base model for semantic analyzer models.

Subclasses should implement only the methods they need: a pure
embedder may implement `embed` but raise `NotImplementedError` for
`predict` and vice versa.
"""

name: str = "base-model"

def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
    self.config = config or {}
    self._loaded = False

def is_loaded(self) -> bool:
    return self._loaded

@abstractmethod
def load(self) -> None:
    """Load model resources (weights, tokenizer, runtime sessions).

    Should be safe to call multiple times (idempotent) and must set
    `self._loaded = True` on success.
    """
    raise NotImplementedError

@abstractmethod
def unload(self) -> None:
    """Free resources held by the model (close sessions, free memory).

    After `unload()` the model may be `load()`-ed again.
    """
    raise NotImplementedError

# --- Optional inference APIs ---
def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
    """Return vector embeddings for the provided texts.

    Default implementation raises NotImplementedError. Subclasses
    that support embeddings should override this method.
    """
    raise NotImplementedError("embed not implemented for this model")

def predict(self, texts: Sequence[str]) -> Sequence[Any]:
    """Return predictions for the provided texts.

    Predictions may be labels, probabilities, or structured dicts.
    """
    raise NotImplementedError("predict not implemented for this model")

# --- Optional persistence helpers ---
def save(self, path: str) -> None:
    """Optional: persist model artifacts to disk.

    Not all model types will implement saving (e.g., transformer
    models delegated to HF hub). Implement if useful.
    """
    raise NotImplementedError("save not implemented for this model")

def info(self) -> Dict[str, Any]:
    """Return a serializable dict describing the model (name, config,
    loaded state, and optionally runtime metadata).
    """
    return {
        "name": self.name,
        "config": self.config,
        "loaded": bool(self._loaded),
    }

Lightweight mixin for async-capable implementations

class AsyncModelMixin: async def aload(self) -> None: """Async load hook. Falls back to load() when not overridden.""" return self.load()

async def aunload(self) -> None:
    """Async unload hook. Falls back to `unload()` when not overridden."""
    return self.unload()

async def aembed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
    """Async embed fallback to sync `embed`."""
    return self.embed(texts)

async def apredict(self, texts: Sequence[str]) -> Sequence[Any]:
    """Async predict fallback to sync `predict`."""
    return self.predict(texts)


