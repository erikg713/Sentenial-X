# core/cortex/ai_core/model_utils.py
# PRODUCTION-READY MODEL UTILITIES v4.0
# Enterprise-grade | Secure | Observable | Zero-trust
# Date: November 06, 2025
# Location: Houston, Texas, US
# Division: Secure Autonomous Systems

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import hashlib
import json
import logging
import traceback
from functools import wraps
from contextlib import contextmanager
import weakref

# ————————————————————————
# Secure Logging
# ————————————————————————
LOG_DIR = Path("logs/model_utils")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ModelUtils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler(LOG_DIR / "utils.log")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())

json_logger = logging.getLogger("ModelUtilsJSON")
json_logger.setLevel(logging.INFO)
if not json_logger.handlers:
    jfh = logging.FileHandler(LOG_DIR / "events.jsonl")
    jfh.setFormatter(logging.Formatter("%(message)s"))
    json_logger.addHandler(jfh)

# ————————————————————————
# Constants & Security Config
# ————————————————————————
SUPPORTED_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
}

DEFAULT_EMBED_DIM = 256
MAX_NORM_THRESHOLD = 15.0
GRAD_CLIP_MAX = 1.0

# ————————————————————————
# Robust Decorator
# ————————————————————————
def secure_model_op(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Model op failed in {func.__name__}: {e}\n{traceback.format_exc()}")
            json_logger.info(json.dumps({
                "event": "model_op_failure",
                "operation": func.__name__,
                "error": str(e),
                "timestamp": time.time()
            }))
            raise
    return wrapper

# ————————————————————————
# Model Fingerprinting & Integrity
# ————————————————————————
class ModelFingerprint:
    """Immutable fingerprint for model integrity verification"""
    def __init__(self, model: nn.Module):
        self.model_id = hashlib.sha256(str(id(model)).encode()).hexdigest()[:16]
        self.param_hash = self._compute_param_hash(model)
        self.structure_hash = self._compute_structure_hash(model)
        self.timestamp = time.time()
        logger.info(f"ModelFingerprint created: {self.model_id}")

    def _compute_param_hash(self, model: nn.Module) -> str:
        hashes = []
        for p in model.parameters():
            if p.data is not None:
                hashes.append(hashlib.sha256(p.data.cpu().numpy().tobytes()).hexdigest())
        return hashlib.sha256("".join(hashes).encode()).hexdigest()

    def _compute_structure_hash(self, model: nn.Module) -> str:
        structure = []
        for name, module in model.named_modules():
            structure.append(f"{name}:{module.__class__.__name__}")
        return hashlib.sha256("\n".join(structure).encode()).hexdigest()

    def verify(self, model: nn.Module) -> bool:
        current_param = self._compute_param_hash(model)
        current_struct = self._compute_structure_hash(model)
        match = (current_param == self.param_hash) and (current_struct == self.structure_hash)
        json_logger.info(json.dumps({
            "event": "fingerprint_verification",
            "model_id": self.model_id,
            "match": match,
            "timestamp": time.time()
        }))
        return match

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "param_hash": self.param_hash,
            "structure_hash": self.structure_hash,
            "timestamp": self.timestamp
        }

# ————————————————————————
# Secure Hidden State Projector (for Brainstem defense)
# ————————————————————————
class DefenseProjector(nn.Module):
    """Projects high-dim hidden states → fixed 256-dim for anomaly detection"""
    def __init__(self, input_dim: int, output_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        self.output_dim = output_dim
        self._fingerprint = None
        self._register_hooks()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:  # (batch, seq, dim)
            hidden_states = hidden_states.mean(dim=1)  # Pool over sequence
        projected = self.projector(hidden_states)
        return projected

    def _register_hooks(self):
        def hook(module, input, output):
            norm = output.norm(p=2, dim=-1).mean().item()
            if norm > MAX_NORM_THRESHOLD:
                logger.warning(f"High-norm projection detected: {norm:.2f}")
                json_logger.info(json.dumps({
                    "event": "high_norm_projection",
                    "norm": norm,
                    "layer": "DefenseProjector"
                }))
        self.projector.register_forward_hook(hook)

    def get_fingerprint(self) -> ModelFingerprint:
        if self._fingerprint is None:
            self._fingerprint = ModelFingerprint(self)
        return self._fingerprint

# ————————————————————————
# Model Utilities Core
# ————————————————————————
class ModelUtils:
    @staticmethod
    @secure_model_op
    def sanitize_weights(model: nn.Module) -> nn.Module:
        """Zero-trust weight sanitization"""
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()
                if torch.any(torch.isnan(p.data)) or torch.any(torch.isinf(p.data)):
                    logger.warning("NaN/Inf detected in weights - clamping")
                    p.data.clamp_(-10.0, 10.0)
        return model

    @staticmethod
    @secure_model_op
    def gradient_clipping(model: nn.Module, max_norm: float = GRAD_CLIP_MAX):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        total_norm = sum(p.grad.norm() for p in model.parameters() if p.grad is not None)
        json_logger.info(json.dumps({
            "event": "gradient_clip",
            "total_norm": float(total_norm),
            "max_norm": max_norm
        }))

    @staticmethod
    @secure_model_op
    def embed_projection(
        hidden_states: torch.Tensor,
        projector: DefenseProjector
    ) -> np.ndarray:
        """Project hidden states for Brainstem analysis"""
        with torch.no_grad():
            projected = projector(hidden_states).cpu().numpy()
        return projected.astype(np.float32)

    @staticmethod
    def export_safetensors(
        model: nn.Module,
        path: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None
    ):
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError("safetensors not installed. pip install safetensors")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = model.state_dict()
        meta = metadata or {}
        meta["fingerprint"] = json.dumps(ModelFingerprint(model).to_dict())
        save_file(state_dict, path, metadata=meta)
        logger.info(f"Model exported safely to {path}")

    @staticmethod
    def load_safetensors(
        path: Union[str, Path],
        device: str = "cpu"
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError("safetensors not installed")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        
        state_dict = load_file(path, device=device)
        metadata = state_dict.get("__metadata__", {})
        # Reconstruct model would go here in real use
        logger.info(f"Model loaded from {path}")
        return state_dict, metadata

    @staticmethod
    @contextmanager
    def inference_mode(model: nn.Module):
        """Strict no-grad + eval mode"""
        model.eval()
        with torch.no_grad(), torch.inference_mode():
            yield model

# ————————————————————————
# Production Demo
# ————————————————————————
def production_demo():
    print("MODEL UTILITIES v4.0 – SECURE AUTONOMOUS SYSTEMS")
    print("Houston, Texas, US | November 06, 2025\n")

    # Dummy model
    class SecureModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4096, 512)
            self.proj = DefenseProjector(512)

        def forward(self, x):
            return self.proj(self.fc(x))

    model = SecureModel()
    utils = ModelUtils()

    # Sanitize
    model = utils.sanitize_weights(model)

    # Fingerprint
    fp = model.proj.get_fingerprint()
    print(f"Model ID: {fp.model_id}")
    print(f"Integrity: {fp.verify(model)}")

    # Simulate hidden state
    hidden = torch.randn(1, 32, 512)
    embedding = utils.embed_projection(hidden, model.proj)
    print(f"Projected embedding shape: {embedding.shape}")
    print(f"Norm: {np.linalg.norm(embedding):.2f}")

if __name__ == "__main__":
    production_demo()
