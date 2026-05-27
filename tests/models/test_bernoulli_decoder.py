from __future__ import annotations

import torch

from mvnjsdm.models.decoders import BernoulliDecoder
from mvnjsdm.models.likelihoods import BernoulliLikelihood


def test_bernoulli_decoder_shapes():
    dec = BernoulliDecoder(in_dim=5, n_features=11)
    z = torch.randn(4, 5)
    params = dec(z)
    assert params.mu.shape == (4, 11)


def test_bernoulli_decoder_loading_matrix():
    dec = BernoulliDecoder(in_dim=4, n_features=7)
    W = dec.loading_matrix()
    assert W.shape == (4, 7)


def test_bernoulli_decoder_training_reduces_loss():
    torch.manual_seed(0)
    in_dim = 4
    F = 8
    N = 200
    W_true = torch.randn(in_dim, F)
    z = torch.randn(N, in_dim)
    logits_true = z @ W_true
    x = (torch.sigmoid(logits_true) > torch.rand(N, F)).float()
    mask = torch.ones_like(x)

    dec = BernoulliDecoder(in_dim=in_dim, n_features=F)
    lik = BernoulliLikelihood()
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
    assert final < initial - 0.5, f"final={final} not less than initial={initial}"
