"""Tests for the neural AR transition."""

from __future__ import annotations

import torch

from mvnjsdm.models.ar import NeuralARTransition


def test_ar_prefers_true_noise_over_high_noise():
    """For a random-walk trajectory with sigma=0.1, the log-prob at the true
    sigma is higher than at sigma=2.0 — the loose model gets too penalised by
    the const term."""
    torch.manual_seed(0)
    n = 24
    K = 3
    t = torch.linspace(0.0, n - 1, n)
    eps = 0.1 * torch.randn(n, K)
    z = torch.zeros(n, K)
    for i in range(1, n):
        z[i] = z[i - 1] + eps[i]
    groups = torch.zeros(n, dtype=torch.long)

    ar_tight = NeuralARTransition(latent_dim=K, noise_scale=0.1)
    ar_loose = NeuralARTransition(latent_dim=K, noise_scale=2.0)
    lp_tight = ar_tight.transition_log_prob(z, t, groups).item()
    lp_loose = ar_loose.transition_log_prob(z, t, groups).item()
    assert lp_tight > lp_loose


def test_ar_multi_group_independence():
    """Two independent trajectories in the same batch shouldn't pollute each
    other: log-prob equals the sum of per-group log-probs computed separately.
    """
    torch.manual_seed(0)
    K = 2
    n_per = 8
    # Group 0: starts near 0
    t0 = torch.arange(n_per, dtype=torch.float32)
    z0 = torch.cumsum(0.1 * torch.randn(n_per, K), dim=0)
    # Group 1: same length, different trajectory
    t1 = torch.arange(n_per, dtype=torch.float32)
    z1 = torch.cumsum(0.1 * torch.randn(n_per, K), dim=0) + 10.0
    z = torch.cat([z0, z1], dim=0)
    t = torch.cat([t0, t1], dim=0)
    groups = torch.cat([torch.zeros(n_per, dtype=torch.long), torch.ones(n_per, dtype=torch.long)])

    ar = NeuralARTransition(latent_dim=K, noise_scale=0.1)
    joint = ar.transition_log_prob(z, t, groups).item()
    lp0 = ar.transition_log_prob(z0, t0, torch.zeros(n_per, dtype=torch.long)).item()
    lp1 = ar.transition_log_prob(z1, t1, torch.zeros(n_per, dtype=torch.long)).item()
    assert abs(joint - (lp0 + lp1)) < 1e-3


def test_ar_handles_singletons():
    """Groups with only one observation contribute zero."""
    ar = NeuralARTransition(latent_dim=2)
    z = torch.zeros(1, 2)
    t = torch.zeros(1)
    g = torch.zeros(1, dtype=torch.long)
    lp = ar.transition_log_prob(z, t, g).item()
    assert lp == 0.0


def test_ar_gradient_flow():
    torch.manual_seed(0)
    ar = NeuralARTransition(latent_dim=2, hidden=8, noise_scale=1.0)
    z = torch.randn(6, 2, requires_grad=False)
    t = torch.linspace(0, 5, 6)
    g = torch.zeros(6, dtype=torch.long)
    lp = ar.transition_log_prob(z, t, g)
    lp.backward()
    # Final-layer params start at zero so we expect them to acquire grads now
    assert ar.net[-1].weight.grad is not None
    assert torch.isfinite(ar.net[-1].weight.grad).all()
