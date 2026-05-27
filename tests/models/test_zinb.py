"""ZINB likelihood + decoder tests."""

from __future__ import annotations

import torch

from mvnjsdm.models.decoders import ZINBDecoder
from mvnjsdm.models.likelihoods import (
    LikelihoodParams,
    NBLikelihood,
    ZINBLikelihood,
)


def _make_zinb_params(B: int, F: int, mu_val: float, theta_val: float, gate_val: float):
    mu = torch.full((B, F), mu_val)
    theta = torch.full((B, F), theta_val)
    gate = torch.full((B, F), gate_val)
    return LikelihoodParams(mu=mu, theta=theta, gate=gate)


def test_zinb_log_prob_finite():
    B, F = 4, 10
    rng = torch.Generator().manual_seed(0)
    x = torch.poisson(torch.full((B, F), 2.0), generator=rng)
    # zero out half to be zero-inflated
    x[:, : F // 2] = 0
    params = _make_zinb_params(B, F, mu_val=2.0, theta_val=5.0, gate_val=0.3)
    lp = ZINBLikelihood().log_prob(x, params)
    assert lp.shape == (B,)
    assert torch.isfinite(lp).all()


def test_zinb_better_than_nb_on_zero_inflated_data():
    """With many zeros beyond NB's expectation, ZINB should give higher
    log-prob than a matched NB (gate=0 case)."""
    B, F = 8, 32
    # Generate data heavily zero-inflated relative to NB(mu=3, theta=5)
    rng = torch.Generator().manual_seed(7)
    x = torch.poisson(torch.full((B, F), 3.0), generator=rng)
    # force 70% zeros
    drop = torch.rand((B, F), generator=rng) < 0.7
    x = torch.where(drop, torch.zeros_like(x), x)

    nb_params = LikelihoodParams(mu=torch.full((B, F), 3.0), theta=torch.full((B, F), 5.0))
    nb_lp = NBLikelihood().log_prob(x, nb_params).sum()

    # ZINB with gate = 0.6 (close to truth)
    zinb_params = _make_zinb_params(B, F, mu_val=3.0, theta_val=5.0, gate_val=0.6)
    zinb_lp = ZINBLikelihood().log_prob(x, zinb_params).sum()

    assert zinb_lp.item() > nb_lp.item()


def test_zinb_decoder_forward_shapes():
    in_dim, F = 4, 12
    dec = ZINBDecoder(in_dim=in_dim, n_features=F, use_size_factor=True)
    z = torch.randn(5, in_dim)
    sf = torch.full((5,), 100.0)
    params = dec(z, size_factor=sf)
    assert params.mu.shape == (5, F)
    assert params.theta is not None and params.theta.shape == (5, F)
    assert params.gate is not None and params.gate.shape == (5, F)
    assert (params.gate >= 0).all() and (params.gate <= 1).all()
    assert dec.loading_matrix().shape == (in_dim, F)


def test_zinb_decoder_end_to_end_logprob():
    in_dim, F = 3, 8
    dec = ZINBDecoder(in_dim=in_dim, n_features=F, use_size_factor=False)
    z = torch.randn(4, in_dim)
    params = dec(z)
    x = torch.zeros(4, F)
    x[:, :2] = 5.0
    lp = ZINBLikelihood().log_prob(x, params)
    assert lp.shape == (4,)
    assert torch.isfinite(lp).all()
