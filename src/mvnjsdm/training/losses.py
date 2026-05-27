"""ELBO assembly.

Loss = sum_a w_a * recon_NLL_a
     + beta_shared * KL_shared
     + sum_a beta_private_a * KL_private_a
     + sum_a feature_prior_NLL_a    # = - log_prob(W_a) of the prior

KL is closed-form against the hierarchical prior (with offsets as context).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..models.likelihoods import Likelihood
from ..models.mvnjsdm import MVNeuralJSDM
from ..models.priors import HierarchicalPrior, StandardNormalPrior


@dataclass
class LossWeights:
    beta_shared: float = 1.0
    beta_private: dict[str, float] | None = None
    recon_weight: dict[str, float] | None = None

    def beta_priv(self, name: str) -> float:
        if self.beta_private is None:
            return 1.0
        return self.beta_private.get(name, 1.0)

    def recon_w(self, name: str) -> float:
        if self.recon_weight is None:
            return 1.0
        return self.recon_weight.get(name, 1.0)


def compute_loss(
    model: MVNeuralJSDM,
    batch: dict[str, Any],
    out: dict[str, Any],
    weights: LossWeights,
) -> dict[str, Tensor]:
    mu_q = out["mu_q"]
    lv_q = out["logvar_q"]
    mu_h = out["mu_h"]
    lv_h = out["logvar_h"]
    decoded = out["decoded"]

    B = mu_q.shape[0]
    device = mu_q.device

    # KL against the hierarchical prior. We split into shared and per-private blocks.
    shared = model.shared_dim
    private_offsets = model.private_offsets
    private_dims = model.private_dims

    has_hierarchy = len(model.hierarchy.level_names) > 0
    sn_prior = StandardNormalPrior()
    h_prior = HierarchicalPrior()

    def _kl(mu_block, lv_block, mu_p_block, lv_p_block):
        if not has_hierarchy:
            return sn_prior.kl_divergence(mu_block, lv_block)
        ctx = {"offset_mu": mu_p_block, "offset_logvar": lv_p_block}
        return h_prior.kl_divergence(mu_block, lv_block, ctx)

    # shared block
    kl_shared = _kl(
        mu_q[:, :shared],
        lv_q[:, :shared],
        mu_h[:, :shared],
        lv_h[:, :shared],
    )

    kl_private_per: dict[str, Tensor] = {}
    for s in model.assay_specs:
        off = private_offsets[s.name]
        d = private_dims[s.name]
        if d == 0:
            kl_private_per[s.name] = torch.zeros(B, device=device)
            continue
        kl_private_per[s.name] = _kl(
            mu_q[:, off : off + d],
            lv_q[:, off : off + d],
            mu_h[:, off : off + d],
            lv_h[:, off : off + d],
        )

    # reconstruction NLL per assay
    recon_per: dict[str, Tensor] = {}
    feat_prior_per: dict[str, Tensor] = {}
    for s in model.assay_specs:
        x = batch["assays"][s.name]["x"]
        mask = batch["assays"][s.name]["mask"]
        # presence mask per sample
        present = (mask.sum(dim=-1) > 0).float()
        likeli: Likelihood = model.likelihoods[s.name]
        lp = likeli.log_prob(x, decoded[s.name], mask=mask)  # [B]
        recon_per[s.name] = -lp * present
        feat_prior_per[s.name] = model.decoders[s.name].feature_prior_loss()

    # totals (mean over batch)
    total = torch.zeros((), device=device)
    parts: dict[str, Tensor] = {
        "kl_shared": kl_shared.mean(),
    }
    total = total + weights.beta_shared * kl_shared.mean()
    for name, kl_p in kl_private_per.items():
        parts[f"kl_priv_{name}"] = kl_p.mean()
        total = total + weights.beta_priv(name) * kl_p.mean()
    for name, rp in recon_per.items():
        parts[f"recon_{name}"] = rp.mean()
        total = total + weights.recon_w(name) * rp.mean()
    for name, fp in feat_prior_per.items():
        parts[f"feature_prior_{name}"] = fp
        # fp is already -log_prob(W); divide by B so it scales similarly
        total = total + fp / max(B, 1)
    parts["loss"] = total
    return parts
