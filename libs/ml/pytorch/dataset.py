from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


def _as_tensor(
    obj: Union[Tensor, Sequence, np.ndarray],
    a44: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Convert `obj` to a torch.Tensor efficiently. If `obj` is already a Tensor
    it is returned (casted if dtype provided).
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(dty
    if dtype is None:
        # Infer a sensible default if dtype not provided
        if np.issubdtype(arr.dtype, np.integer):
            dtype = torch.long
        else:
            dtype = torch.float32
    # torch.as_tensor will avoid copying when possible
    return torch.as_tensor(arr, dtype=dtype)


class ThreatSampleDataset(Dataset):
    """
    Lightweight dataset wrapper for features X and labels y.

    Improvements over a minimal implementation:
    - Accepts numpy arrays, lists or torch.Tensor inputs without unnecessary copies.
    - Infers sensible dtypes (float32 for inputs, long for integer targets).
    - Optional transforms for inputs and targets.
    - Optional return_index flag to return the sample index (useful for debugging / diagnostics).
    - Input validation with helpful error messages.
    """

    def __init__(
        self,
        X: Union[Sequence, np.ndarray, Tensor],
        y: Union[Sequence, np.ndarray, Tensor],
        *,
        transform: Optional[Callable[[Tensor], Any]] = None,
        target_transform: Optional[Callable[[Tensor], Any]] = None,
        return_index: bool = False,
    ) -> None:
        # Convert features to float32 tensors (no extra copy when possible)
        self.X: Tensor = _as_tensor(X, dtype=torch.float32)

        # Convert targets; infer integer vs float automatically
        # If y is a 1-d integer-like array, keep as long for classification
        if isinstance(y, torch.Tensor):
            inferred_target_dtype = torch.long if y.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) else torch.float32
        else:
            arr = np.asarray(y)
            inferred_target_dtype = torch.long if np.issubdtype(arr.dtype, np.integer) else torch.float32

        self.y: Tensor = _as_tensor(y, dtype=inferred_target_dtype)

        if len(self.X) != len(self.y):
            raise ValueError(f"Length mismatch: len(X)={len(self.X)} vs len(y)={len(self.y)}")

        self.transform = transform
        self.target_transform = target_transform
        self.return_index = bool(return_index)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, int]]:
        x = self.X[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)

        if self.target_transform is not None:
            y = self.target_transform(y)

        if self.return_index:
            return x, y, int(idx)
        return x, y

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_samples={len(self)}, "
            f"feature_shape={tuple(self.X.shape[1:]) if self.X.dim() > 1 else (self.X.shape[0],)}, "
            f"feature_dtype={self.X.dtype}, target_dtype={self.y.dtype})"
        )
