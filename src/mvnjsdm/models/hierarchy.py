"""Hierarchical random effects on the joint latent z.

Each configured level contributes an additive (mu, logvar) offset on the
joint z. Three modes per level:

* ``hierarchical_prior`` -- proper partial pooling: learn a per-group
  random-effect mean (in z space) and a learnable global log-variance for the
  level. Implemented as an nn.Embedding for means plus a scalar logvar.
* ``embedding`` -- simple nn.Embedding added to mu, logvar contribution = 0.
* ``off`` -- nothing.

Combination across levels: sum of mu contributions; logvar = log(sum exp(lv_l))
(i.e. variances add).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import LongTensor, Tensor


@dataclass
class LevelSpec:
    name: str
    n_groups: int
    mode: str = "hierarchical_prior"  # 'hierarchical_prior' | 'embedding' | 'off'
    init_scale: float = 0.1


class _Level(nn.Module):
    def __init__(self, spec: LevelSpec, K: int) -> None:
        super().__init__()
        self.spec = spec
        self.K = K
        if spec.mode == "off":
            self.emb = None
            self.raw_logvar = None
        elif spec.mode == "embedding":
            self.emb = nn.Embedding(spec.n_groups, K)
            nn.init.normal_(self.emb.weight, std=spec.init_scale)
            self.raw_logvar = None
        elif spec.mode == "hierarchical_prior":
            self.emb = nn.Embedding(spec.n_groups, K)
            nn.init.normal_(self.emb.weight, std=spec.init_scale)
            # global log-variance per latent dim, learnable
            self.raw_logvar = nn.Parameter(torch.full((K,), -2.0))
        else:
            raise ValueError(f"unknown hierarchy mode {spec.mode}")

    def forward(self, ids: LongTensor) -> tuple[Tensor, Tensor]:
        if self.emb is None:
            B = ids.shape[0]
            zeros = torch.zeros(B, self.K, device=ids.device)
            return zeros, torch.full_like(zeros, -50.0)  # ~0 variance
        mu = self.emb(ids)
        if self.raw_logvar is None:
            lv = torch.full_like(mu, -50.0)
        else:
            lv = self.raw_logvar.unsqueeze(0).expand_as(mu)
        return mu, lv


class HierarchyBlock(nn.Module):
    """Sums random-effect offsets across configured levels."""

    def __init__(self, levels: list[LevelSpec], K_joint: int) -> None:
        super().__init__()
        self.K = K_joint
        self.levels = nn.ModuleDict(
            {spec.name: _Level(spec, K_joint) for spec in levels}
        )
        self.level_names = [s.name for s in levels]

    def forward(self, group_ids: dict[str, LongTensor]) -> tuple[Tensor, Tensor]:
        if not self.level_names:
            # use shape from any group id if available; otherwise raise
            raise ValueError("HierarchyBlock has no levels configured")
        # determine batch size from one input
        sample = next(iter(group_ids.values()))
        B = sample.shape[0]
        mu_sum = torch.zeros(B, self.K, device=sample.device)
        var_sum = torch.zeros(B, self.K, device=sample.device)
        for name in self.level_names:
            ids = group_ids[name]
            mu_l, lv_l = self.levels[name](ids)
            mu_sum = mu_sum + mu_l
            var_sum = var_sum + lv_l.exp()
        lv_sum = torch.log(var_sum.clamp_min(1e-12))
        return mu_sum, lv_sum


def empty_hierarchy(K: int) -> HierarchyBlock:
    """A HierarchyBlock with no levels (returns zero offsets)."""
    block = HierarchyBlock.__new__(HierarchyBlock)
    nn.Module.__init__(block)
    block.K = K
    block.levels = nn.ModuleDict()
    block.level_names = []

    def forward(group_ids):
        if group_ids:
            sample = next(iter(group_ids.values()))
            B = sample.shape[0]
            device = sample.device
        else:
            B = 1
            device = torch.device("cpu")
        mu = torch.zeros(B, K, device=device)
        return mu, torch.full_like(mu, -50.0)

    block.forward = forward  # type: ignore
    return block
