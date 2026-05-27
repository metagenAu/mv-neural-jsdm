from __future__ import annotations

import torch

from mvnjsdm.models.decoders import NBDecoder


def test_nb_decoder_shapes():
    dec = NBDecoder(in_dim=8, n_features=15, use_size_factor=True)
    z = torch.randn(4, 8)
    sf = torch.full((4,), 1000.0)
    out = dec(z, size_factor=sf)
    assert out.mu.shape == (4, 15)
    assert out.theta.shape == (4, 15)
    assert torch.all(out.mu > 0)
    assert torch.all(out.theta > 0)


def test_nb_decoder_feature_prior_loss_is_zero_when_unset():
    dec = NBDecoder(in_dim=8, n_features=15)
    val = dec.feature_prior_loss()
    assert val.item() == 0.0
