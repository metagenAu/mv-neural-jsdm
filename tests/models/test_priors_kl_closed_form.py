from __future__ import annotations

import torch

from mvnjsdm.models.priors import HierarchicalPrior, StandardNormalPrior


def test_standard_normal_kl_matches_formula():
    mu = torch.randn(5, 4)
    lv = torch.randn(5, 4) * 0.1
    kl = StandardNormalPrior().kl_divergence(mu, lv)
    # KL(N(mu, sigma^2) || N(0,1)) = 0.5 * (mu^2 + var - 1 - logvar)
    expected = 0.5 * (mu.pow(2) + lv.exp() - 1.0 - lv).sum(dim=-1)
    assert torch.allclose(kl, expected, atol=1e-6)


def test_hierarchical_prior_zero_offset_equals_standard():
    mu = torch.randn(3, 4)
    lv = torch.randn(3, 4) * 0.05
    sn = StandardNormalPrior().kl_divergence(mu, lv)
    ctx = {"offset_mu": torch.zeros_like(mu), "offset_logvar": torch.zeros_like(lv)}
    hp = HierarchicalPrior().kl_divergence(mu, lv, ctx)
    assert torch.allclose(sn, hp, atol=1e-6)
