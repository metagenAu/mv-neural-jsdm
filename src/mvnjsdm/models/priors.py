"""Latent priors over the joint z space.

All priors expose:

    log_prob(z, context) -> [B]
    kl_divergence(mu, logvar, context) -> [B]

Closed-form KL is used where available.
"""

from __future__ import annotations

import abc
import math

import torch
import torch.nn as nn
from torch import Tensor


class LatentPrior(nn.Module, abc.ABC):
    @abc.abstractmethod
    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor: ...

    @abc.abstractmethod
    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor: ...


class StandardNormalPrior(LatentPrior):
    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor:
        return (-0.5 * (z ** 2 + math.log(2 * math.pi))).sum(dim=-1)

    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor:
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)


class HierarchicalPrior(LatentPrior):
    """Random-effect prior whose mean is shifted by hierarchy offsets.

    Context expected to contain ``offset_mu`` and ``offset_logvar`` from a
    HierarchyBlock; the prior becomes N(offset_mu, exp(offset_logvar)).
    """

    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor:
        if context is None or "offset_mu" not in context:
            mu_p = torch.zeros_like(z)
            lv_p = torch.zeros_like(z)
        else:
            mu_p = context["offset_mu"]
            lv_p = context["offset_logvar"]
        var_p = lv_p.exp()
        return (-0.5 * ((z - mu_p) ** 2 / var_p + lv_p + math.log(2 * math.pi))).sum(dim=-1)

    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor:
        if context is None or "offset_mu" not in context:
            mu_p = torch.zeros_like(mu)
            lv_p = torch.zeros_like(logvar)
        else:
            mu_p = context["offset_mu"]
            lv_p = context["offset_logvar"]
        var_p = lv_p.exp()
        var_q = logvar.exp()
        kl = 0.5 * (
            (var_q + (mu - mu_p).pow(2)) / var_p
            - 1.0
            + lv_p
            - logvar
        )
        return kl.sum(dim=-1)


class GPLatentPrior(LatentPrior):  # skeleton
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def log_prob(self, z, context=None):  # pragma: no cover
        raise NotImplementedError("GPLatentPrior is a skeleton")

    def kl_divergence(self, mu, logvar, context=None):  # pragma: no cover
        raise NotImplementedError("GPLatentPrior is a skeleton")


class CompositePrior(LatentPrior):
    """Sum of constituent priors' log-probs and KLs."""

    def __init__(self, priors: list[LatentPrior]) -> None:
        super().__init__()
        self.priors = nn.ModuleList(priors)

    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor:
        return torch.stack([p.log_prob(z, context) for p in self.priors], dim=0).sum(dim=0)

    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor:
        return torch.stack(
            [p.kl_divergence(mu, logvar, context) for p in self.priors], dim=0
        ).sum(dim=0)
