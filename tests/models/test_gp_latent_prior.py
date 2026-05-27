"""Tests for the GP latent prior."""

from __future__ import annotations

import math

import pytest
import torch

from mvnjsdm.models._kernels import KERNELS, get_kernel
from mvnjsdm.models.priors import GPLatentPrior


@pytest.mark.parametrize("name", ["rbf", "matern_3_2", "matern_5_2"])
def test_kernel_symmetric_and_psd(name):
    fn = get_kernel(name)
    torch.manual_seed(0)
    x = torch.randn(10, 1)
    ls = torch.tensor([1.0])
    os_ = torch.tensor(1.0)
    K = fn(x, x, ls, os_)
    assert K.shape == (10, 10)
    assert torch.allclose(K, K.T, atol=1e-5)
    eigs = torch.linalg.eigvalsh(K + 1e-6 * torch.eye(10))
    assert eigs.min().item() > -1e-5


def test_periodic_kernel_periodicity():
    fn = get_kernel("periodic")
    x1 = torch.tensor([[0.0]])
    x2 = torch.tensor([[1.0]])
    ls = torch.tensor([1.0])
    os_ = torch.tensor(1.0)
    period = torch.tensor([1.0])
    K01 = fn(x1, x2, ls, os_, period).item()
    K00 = fn(x1, x1, ls, os_, period).item()
    # at one full period apart, K should equal K(x,x)
    assert abs(K01 - K00) < 1e-5


def test_gp_log_prob_prefers_smooth_over_rough():
    torch.manual_seed(0)
    B = 32
    t = torch.linspace(-3.0, 3.0, B).unsqueeze(-1)
    z_smooth = torch.stack([torch.sin(t.squeeze()), torch.cos(t.squeeze())], dim=-1)
    z_rough = torch.randn(B, 2) * 1.0

    prior = GPLatentPrior(n_dims=2, input_dim=1, kernel="rbf", init_lengthscale=1.0)
    lp_smooth = prior.log_prob(z_smooth, {"indices": t}).sum().item()
    lp_rough = prior.log_prob(z_rough, {"indices": t}).sum().item()
    assert lp_smooth > lp_rough


def test_gp_log_prob_gradients_flow():
    torch.manual_seed(0)
    B = 16
    t = torch.linspace(0.0, 5.0, B).unsqueeze(-1)
    prior = GPLatentPrior(n_dims=1, input_dim=1, kernel="rbf")
    z = torch.sin(t).detach()
    lp = prior.log_prob(z, {"indices": t}).sum()
    lp.backward()
    assert prior.raw_log_lengthscale.grad is not None
    assert prior.raw_log_outputscale.grad is not None
    assert torch.isfinite(prior.raw_log_lengthscale.grad).all()
    assert torch.isfinite(prior.raw_log_outputscale.grad).all()


def test_gp_kl_divergence_nonneg_and_finite():
    torch.manual_seed(0)
    B = 12
    t = torch.linspace(0.0, 4.0, B).unsqueeze(-1)
    prior = GPLatentPrior(n_dims=2, input_dim=1, kernel="rbf")
    mu = 0.1 * torch.randn(B, 2)
    logvar = torch.full((B, 2), -2.0)
    kl = prior.kl_divergence(mu, logvar, {"indices": t})
    assert kl.shape == (B,)
    assert torch.isfinite(kl).all()


def test_gp_periodic_kernel_smooth_input():
    torch.manual_seed(0)
    B = 30
    t = torch.linspace(0.0, 4 * math.pi, B).unsqueeze(-1)
    # Periodic signal aligned with period 2pi
    z = torch.sin(t)
    prior = GPLatentPrior(
        n_dims=1, input_dim=1, kernel="periodic",
        init_lengthscale=1.0, init_period=2 * math.pi,
    )
    lp = prior.log_prob(z, {"indices": t}).sum().item()
    assert math.isfinite(lp)
