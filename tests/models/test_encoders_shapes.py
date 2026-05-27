from __future__ import annotations

import torch

from mvnjsdm.models.encoders import CountEncoder, EnvCovariateEncoder


def test_count_encoder_shape():
    enc = CountEncoder(n_features=20, out_dim=8)
    x = torch.randint(0, 5, (4, 20)).float()
    mask = torch.ones_like(x)
    sf = torch.full((4,), 100.0)
    mu, lv = enc(x, mask, size_factor=sf)
    assert mu.shape == (4, 8)
    assert lv.shape == (4, 8)


def test_env_encoder_none():
    e = EnvCovariateEncoder(env_dim=0)
    assert e(None) is None
    assert e.out_dim == 0


def test_env_encoder_shape():
    e = EnvCovariateEncoder(env_dim=3, hidden=8, out_dim=5)
    out = e(torch.randn(2, 3))
    assert out.shape == (2, 5)
