"""Data transforms used by encoders and preprocessing.

All functions are NumPy/Torch dual-friendly; we operate on numpy where loading
from disk and on torch tensors inside the model.
"""

from __future__ import annotations

import numpy as np


def clr(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centered log-ratio transform along the last axis.

    x must be non-negative; zeros are clipped to eps.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.where(x > 0, x, eps)
    log_x = np.log(x)
    gm = log_x.mean(axis=-1, keepdims=True)
    return log_x - gm


def log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(x, dtype=np.float64))


def library_size(counts: np.ndarray) -> np.ndarray:
    """Sum across the feature axis (last axis). Used as size factor."""
    return np.asarray(counts).sum(axis=-1)


def zscore_masked(x: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-feature z-score using only mask==1 entries.

    Inputs are 2D [N, F]; mask is 0/1 of the same shape.
    """
    x = np.asarray(x, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    cnt = mask.sum(axis=0).clip(min=1.0)
    mu = (x * mask).sum(axis=0) / cnt
    var = ((x - mu) ** 2 * mask).sum(axis=0) / cnt
    sigma = np.sqrt(var + eps)
    out = (x - mu) / sigma
    return out * mask
