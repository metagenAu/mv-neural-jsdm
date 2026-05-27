from __future__ import annotations

import torch

from mvnjsdm.models.decoders import GaussianDecoder
from mvnjsdm.models.likelihoods import GaussianMaskedLikelihood


def test_gaussian_decoder_shapes():
    dec = GaussianDecoder(in_dim=6, n_features=12)
    z = torch.randn(5, 6)
    params = dec(z)
    assert params.mu.shape == (5, 12)
    assert params.log_sigma is not None
    assert params.log_sigma.shape == (5, 12)


def test_gaussian_decoder_loading_matrix():
    dec = GaussianDecoder(in_dim=4, n_features=10)
    W = dec.loading_matrix()
    assert W.shape == (4, 10)


def test_gaussian_decoder_training_reduces_loss():
    torch.manual_seed(0)
    in_dim = 4
    F = 8
    N = 200
    # synthesise data with known linear loadings
    W_true = torch.randn(in_dim, F)
    z = torch.randn(N, in_dim)
    x = z @ W_true + 0.1 * torch.randn(N, F)
    mask = torch.ones_like(x)

    dec = GaussianDecoder(in_dim=in_dim, n_features=F)
    lik = GaussianMaskedLikelihood()
    opt = torch.optim.Adam(dec.parameters(), lr=5e-2)

    def loss_fn():
        params = dec(z)
        return -lik.log_prob(x, params, mask=mask).mean()

    initial = loss_fn().item()
    for _ in range(60):
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        opt.step()
    final = loss_fn().item()
    assert final < initial - 1.0, f"final={final} not less than initial={initial}"
