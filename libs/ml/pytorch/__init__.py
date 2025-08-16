"""
Sentenial-X ML • PyTorch Backend
--------------------------------
Utilities and convenience APIs for working with PyTorch models inside Sentenial-X.

Features
- Safe/optional import of torch (so non-PyTorch installs still import the package).
- Device management (CUDA, MPS, CPU), dtype selection, and autocast context.
- Deterministic seeding across torch, numpy, and random.
- Model save/load (state_dict-based), with strict/non-strict options.
- Parameter utilities: count, freeze/unfreeze, move_to_device.
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
from dataclasses import dataclass
from typing import Generator, Iterable, Optional, Tuple

logger = logging.getLogger("sentenial.ml.pytorch")
if not logger.handlers:
    _h = logging.StreamHandler()
    _f = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    _h.setFormatter(_f)
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ---- Optional torch import ----
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not available: %s", e)

# ---- Public API ----

__all__ = [
    "is_available",
    "get_device",
    "best_dtype",
    "autocast",
    "set_seed",
    "save_state_dict",
    "load_state_dict",
    "move_to_device",
    "count_parameters",
    "freeze_module",
    "unfreeze_module",
    "torch",
    "DeviceInfo",
]

@dataclass(frozen=True)
class DeviceInfo:
    device: str          # "cuda", "mps", or "cpu"
    index: Optional[int] # GPU index if applicable
    dtype: Optional[str] # "float16", "bfloat16", or "float32"

def is_available() -> bool:
    """Return True if PyTorch is importable."""
    return _TORCH_AVAILABLE

def get_device(prefer: Optional[str] = None, index: Optional[int] = None) -> Tuple["torch.device", DeviceInfo]:
    """
    Choose the best runtime device.

    prefer: one of {"cuda","mps","cpu"} or None to auto-pick.
    index:  specific GPU index for CUDA.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment.")

    prefer = (prefer or os.getenv("SENTENIAL_TORCH_DEVICE") or "").lower() or None

    if prefer == "cuda" or (prefer is None and torch.cuda.is_available()):
        idx = 0 if index is None else index
        dev = torch.device(f"cuda:{idx}")
        info = DeviceInfo("cuda", idx, None)
        return dev, info

    if prefer == "mps" or (prefer is None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        dev = torch.device("mps")
        info = DeviceInfo("mps", None, None)
        return dev, info

    dev = torch.device("cpu")
    info = DeviceInfo("cpu", None, None)
    return dev, info

def best_dtype(prefer: Optional[str] = None) -> "torch.dtype":
    """
    Pick a reasonable dtype based on hardware.
    prefer: {"fp16","bf16","fp32"} (case-insensitive).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment.")

    mapping = {
        "fp16": torch.float16,
        "half": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "full": torch.float32,
    }
    if prefer:
        key = prefer.lower()
        if key in mapping:
            return mapping[key]

    # Heuristic: bf16 on newer GPUs/CPUs if available; else fp16 if CUDA; else fp32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    # Apple M-series (MPS) prefers float32 today; bf16 support varies by OS/torch
    return torch.float32

@contextlib.contextmanager
def autocast(
    device_type: Optional[str] = None,
    dtype: Optional["torch.dtype"] = None,
    enabled: Optional[bool] = None,
) -> Generator[None, None, None]:
    """
    Context manager for mixed precision autocast.

    device_type: {"cuda","cpu","mps"} or None -> auto from get_device()
    dtype: torch.float16 / torch.bfloat16 / torch.float32
    enabled: force enable/disable; if None, enable only when beneficial
    """
    if not _TORCH_AVAILABLE:
        # No-op if torch missing
        yield
        return

    if device_type is None:
        dev, info = get_device()
        device_type = info.device

    if dtype is None:
        dtype = best_dtype()

    # Enable by default only if reduced precision makes sense
    if enabled is None:
        enabled = (device_type in {"cuda"} and dtype in {torch.float16, torch.bfloat16})

    if hasattr(torch, "autocast"):
        with torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
            yield
    else:  # pragma: no cover
        yield

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set global RNG seeds for reproducibility.
    """
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass

    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True)  # torch >= 1.8
            except Exception:
                pass
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def save_state_dict(model: "torch.nn.Module", path: str) -> None:
    """
    Save a model's state_dict to `path`.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment.")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info("Saved state_dict -> %s", path)

def load_state_dict(
    model: "torch.nn.Module",
    path: str,
    strict: bool = True,
    map_location: Optional[str] = None,
) -> "torch.nn.Module":
    """
    Load state_dict from `path` into `model`.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment.")

    if map_location is None:
        dev, _ = get_device()
        map_location = str(dev)

    state = torch.load(path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state, strict=strict)
    if strict:
        logger.info("Loaded state_dict (strict) from %s", path)
    else:
        logger.warning("Loaded state_dict (non-strict) from %s; missing=%s unexpected=%s", path, missing, unexpected)
    return model

def move_to_device(module: "torch.nn.Module", device: Optional["torch.device"] = None) -> "torch.nn.Module":
    """
    Move module to the selected device.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment.")
    if device is None:
        device, _ = get_device()
    module.to(device)
    return module

def count_parameters(module: "torch.nn.Module", trainable_only: bool = True) -> int:
    """
    Count parameters (optionally only trainable ones).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment.")
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())

def freeze_module(module: "torch.nn.Module", names: Optional[Iterable[str]] = None) -> None:
    """
    Freeze all params (or by name subset) by setting requires_grad=False.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment.")

    if names is None:
        for p in module.parameters():
            p.requires_grad = False
        return

    name_set = set(names)
    for n, p in module.named_parameters():
        if any(n == t or n.startswith(f"{t}.") for t in name_set):
            p.requires_grad = False

def unfreeze_module(module: "torch.nn.Module", names: Optional[Iterable[str]] = None) -> None:
    """
    Unfreeze all params (or by name subset) by setting requires_grad=True.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed in this environment.")

    if names is None:
        for p in module.parameters():
            p.requires_grad = True
        return

    name_set = set(names)
    for n, p in module.named_parameters():
        if any(n == t or n.startswith(f"{t}.") for t in name_set):
            p.requires_grad = True
