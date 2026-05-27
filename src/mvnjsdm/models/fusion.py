"""Fusion of per-assay expert distributions into a single (mu, logvar).

Implementations: PoE, MoPoE, Concat.

``ExpertOutput``: per-assay Gaussian factor over the joint latent. The
fusion handles missing assays via ``present_mask``: experts with mask==0 are
replaced by the prior (mu=0, logvar=0 -> variance 1) for that sample.

PoE: product of Gaussian experts including the standard-normal prior. For
diagonal Gaussians, the precision-weighted mean and summed precision give the
joint Gaussian in closed form.

MoPoE: in training, sample one non-empty subset of present experts per
minibatch (this implementation samples one subset shared across the batch).
Inference may average over subsets externally.

Concat: stacks expert means and average their logvars (a non-probabilistic
baseline used only for debugging).
"""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ExpertOutput:
    mu: Tensor  # [B, K]
    logvar: Tensor  # [B, K]
    present_mask: Tensor  # [B], bool


class Fusion(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, experts: dict[str, ExpertOutput]) -> tuple[Tensor, Tensor]: ...


def _poe_combine(
    mus: list[Tensor], logvars: list[Tensor], masks: list[Tensor]
) -> tuple[Tensor, Tensor]:
    """Combine a list of Gaussian experts via product-of-experts, with prior.

    Precisions add. The standard-normal prior contributes prec=1, mean=0 once.
    Experts with mask==0 (absent for that sample) contribute nothing.
    Logvars are clamped to keep precisions finite.
    """
    B, K = mus[0].shape
    device = mus[0].device
    dtype = mus[0].dtype
    sum_prec = torch.ones(B, K, device=device, dtype=dtype)  # prior precision = 1
    sum_pm = torch.zeros(B, K, device=device, dtype=dtype)  # prior mean = 0

    for mu, lv, m in zip(mus, logvars, masks):
        # clamp logvar for stability (precision in [exp(-6), exp(6)])
        lv = lv.clamp(min=-6.0, max=6.0)
        prec = torch.exp(-lv)
        m_ = m.float().unsqueeze(-1)
        # absent samples contribute 0 (prior already accounts for them)
        sum_prec = sum_prec + m_ * prec
        sum_pm = sum_pm + m_ * prec * mu

    sum_prec = sum_prec.clamp_min(1e-6)
    mu_joint = sum_pm / sum_prec
    var_joint = 1.0 / sum_prec
    logvar_joint = torch.log(var_joint.clamp_min(1e-8))
    return mu_joint, logvar_joint


class PoE(Fusion):
    def forward(self, experts: dict[str, ExpertOutput]) -> tuple[Tensor, Tensor]:
        mus = [e.mu for e in experts.values()]
        lvs = [e.logvar for e in experts.values()]
        ms = [e.present_mask for e in experts.values()]
        return _poe_combine(mus, lvs, ms)


class Concat(Fusion):
    """Mean of expert means, averaged logvar. Debug-only."""

    def forward(self, experts: dict[str, ExpertOutput]) -> tuple[Tensor, Tensor]:
        mus = torch.stack([e.mu for e in experts.values()], dim=0)
        lvs = torch.stack([e.logvar for e in experts.values()], dim=0)
        ms = torch.stack([e.present_mask.float() for e in experts.values()], dim=0)
        # weighted mean
        w = ms.unsqueeze(-1)
        denom = w.sum(0).clamp_min(1.0)
        mu = (mus * w).sum(0) / denom
        lv = (lvs * w).sum(0) / denom
        return mu, lv


class MoPoE(Fusion):
    """Mixture over PoE-subsets of present experts (sample one subset per call).

    For this pass we sample one non-empty subset of currently-present experts
    uniformly per minibatch. The chosen subset's PoE result is returned.
    The prior is always included as an expert.
    """

    def __init__(self, seed: int | None = None) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def forward(self, experts: dict[str, ExpertOutput]) -> tuple[Tensor, Tensor]:
        names = list(experts.keys())
        # "present at all" = any sample has the assay
        candidates = [n for n in names if experts[n].present_mask.any().item()]
        if not candidates:
            # fall back to prior
            B, K = experts[names[0]].mu.shape
            device = experts[names[0]].mu.device
            return (
                torch.zeros(B, K, device=device),
                torch.zeros(B, K, device=device),
            )

        if self.training:
            # uniform sample over non-empty subsets
            r = self._rng.randint(1, (1 << len(candidates)) - 1)
            subset = [candidates[i] for i in range(len(candidates)) if (r >> i) & 1]
        else:
            subset = candidates  # inference: full PoE over present experts

        mus = [experts[n].mu for n in subset]
        lvs = [experts[n].logvar for n in subset]
        ms = [experts[n].present_mask for n in subset]
        return _poe_combine(mus, lvs, ms)


def build_fusion(name: str) -> Fusion:
    name = name.lower()
    if name == "poe":
        return PoE()
    if name == "mopoe":
        return MoPoE()
    if name == "concat":
        return Concat()
    raise ValueError(f"unknown fusion: {name}")
