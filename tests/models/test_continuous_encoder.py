from __future__ import annotations

import torch

from mvnjsdm.models.encoders import BinaryEncoder, ContinuousEncoder


def test_continuous_encoder_shape():
    enc = ContinuousEncoder(n_features=12, out_dim=6)
    x = torch.randn(3, 12)
    mask = torch.ones_like(x)
    mu, lv = enc(x, mask)
    assert mu.shape == (3, 6)
    assert lv.shape == (3, 6)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(lv).all()


def test_continuous_encoder_zero_mask_no_nan():
    enc = ContinuousEncoder(n_features=12, out_dim=6)
    x = torch.randn(2, 12)
    mask = torch.zeros_like(x)
    mu, lv = enc(x, mask)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(lv).all()


def test_continuous_encoder_env():
    enc = ContinuousEncoder(n_features=8, out_dim=4, env_dim=3)
    x = torch.randn(2, 8)
    mask = torch.ones_like(x)
    env = torch.randn(2, 3)
    mu, lv = enc(x, mask, batch_cov=env)
    assert mu.shape == (2, 4)


def test_binary_encoder_shape():
    enc = BinaryEncoder(n_features=10, out_dim=5)
    x = torch.randint(0, 2, (4, 10)).float()
    mask = torch.ones_like(x)
    mu, lv = enc(x, mask)
    assert mu.shape == (4, 5)
    assert lv.shape == (4, 5)
    assert torch.isfinite(mu).all()
