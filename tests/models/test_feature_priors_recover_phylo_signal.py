"""The phylogenetic prior should assign higher log-prob to phylogenetically-
structured loadings than to IID loadings of the same scale."""

from __future__ import annotations

import numpy as np
import torch

from mvnjsdm.models.feature_priors import (
    NoFeaturePrior,
    PhylogeneticBrownian,
    PhylogeneticPagel,
)


def _make_phylo(F: int = 16, sigma: float = 0.4, lam: float = 0.9):
    rng = np.random.default_rng(0)
    # synthetic tree covariance: random PSD with strong off-diagonal block
    A = rng.standard_normal((F, F))
    C = A @ A.T / F + np.eye(F) * 0.05
    # draw structured W
    Sigma = sigma**2 * (lam * C + (1 - lam) * np.eye(F))
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(F))
    K = 4
    W_struct = (L @ rng.standard_normal((F, K))).T
    W_iid = rng.standard_normal((K, F)) * sigma
    return C, torch.tensor(W_struct, dtype=torch.float32), torch.tensor(W_iid, dtype=torch.float32)


def test_pagel_prefers_structured_over_iid():
    C, W_struct, W_iid = _make_phylo()
    prior = PhylogeneticPagel(C, init_lambda=0.9)
    # Set log_sigma2 such that scale matches
    with torch.no_grad():
        prior.raw_sigma2.fill_(float(np.log(np.expm1(0.4**2))))
    lp_struct = prior.log_prob(W_struct).item()
    lp_iid = prior.log_prob(W_iid).item()
    assert lp_struct > lp_iid


def test_brownian_runs():
    C, W_struct, _ = _make_phylo(lam=1.0)
    prior = PhylogeneticBrownian(C)
    val = prior.log_prob(W_struct)
    assert torch.isfinite(val)


def test_no_feature_prior():
    p = NoFeaturePrior()
    W = torch.randn(3, 8)
    assert p.log_prob(W).item() == 0.0
