"""Hand-rolled kernel functions used by :class:`GPLatentPrior`.

All kernels return symmetric PSD (modulo jitter) matrices of shape ``[B1, B2]``.
We avoid pulling in GPyTorch as a runtime dependency — only ``torch.cdist`` and
elementwise ops are used. See ``docs/architecture.md`` for the rationale.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _pairwise_distance(x1: Tensor, x2: Tensor, lengthscale: Tensor) -> Tensor:
    """Scaled euclidean pairwise distance ``[B1, B2]``.

    ``lengthscale`` is broadcastable to the input-dim of ``x1``/``x2``.
    """
    # x1, x2: [B, d]
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)
    ls = lengthscale.clamp_min(1e-6)
    x1s = x1 / ls
    x2s = x2 / ls
    return torch.cdist(x1s, x2s, p=2)


def rbf_kernel(
    x1: Tensor, x2: Tensor, lengthscale: Tensor, outputscale: Tensor
) -> Tensor:
    r = _pairwise_distance(x1, x2, lengthscale)
    return outputscale * torch.exp(-0.5 * r.pow(2))


def matern32_kernel(
    x1: Tensor, x2: Tensor, lengthscale: Tensor, outputscale: Tensor
) -> Tensor:
    r = _pairwise_distance(x1, x2, lengthscale)
    sqrt3 = math.sqrt(3.0)
    return outputscale * (1.0 + sqrt3 * r) * torch.exp(-sqrt3 * r)


def matern52_kernel(
    x1: Tensor, x2: Tensor, lengthscale: Tensor, outputscale: Tensor
) -> Tensor:
    r = _pairwise_distance(x1, x2, lengthscale)
    sqrt5 = math.sqrt(5.0)
    return outputscale * (1.0 + sqrt5 * r + (5.0 / 3.0) * r.pow(2)) * torch.exp(-sqrt5 * r)


def periodic_kernel(
    x1: Tensor,
    x2: Tensor,
    lengthscale: Tensor,
    outputscale: Tensor,
    period: Tensor,
) -> Tensor:
    """Periodic kernel (sum across input dims if ``x`` is multi-dim).

    k(x, x') = sigma^2 * exp(-2 sum_d sin^2(pi |x_d - x'_d| / period_d) / ls^2)
    """
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)
    # diff: [B1, B2, d]
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    p = period.clamp_min(1e-6)
    sinarg = math.pi * diff.abs() / p
    s = torch.sin(sinarg).pow(2)
    ls = lengthscale.clamp_min(1e-6)
    expo = -2.0 * (s / ls.pow(2)).sum(dim=-1)
    return outputscale * torch.exp(expo)


KERNELS = {
    "rbf": rbf_kernel,
    "matern_3_2": matern32_kernel,
    "matern32": matern32_kernel,
    "matern_5_2": matern52_kernel,
    "matern52": matern52_kernel,
    "periodic": periodic_kernel,
}


def get_kernel(name: str):
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}; choose from {sorted(KERNELS)}")
    return KERNELS[name]
