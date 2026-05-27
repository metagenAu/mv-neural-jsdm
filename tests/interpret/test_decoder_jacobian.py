from __future__ import annotations

import pytest
import torch

from mvnjsdm.interpret.decoder_jacobian import (
    decoder_jacobian,
    decoder_jacobian_dataset,
)
from mvnjsdm.models.decoders import NBDecoder


def test_nb_decoder_jacobian_matches_analytical():
    torch.manual_seed(0)
    in_dim = 4
    F = 6
    dec = NBDecoder(in_dim=in_dim, n_features=F, use_size_factor=True)

    # build a tiny wrapper "model" so decoder_jacobian's _unwrap path
    # can find this decoder. We'll construct a minimal duck-typed model.
    class _MiniModel:
        def __init__(self, dec):
            from mvnjsdm.models.likelihoods import NBLikelihood
            self.decoders = {"a": dec}
            self.likelihoods = {"a": NBLikelihood()}

    m = _MiniModel(dec)

    z_ref = torch.randn(in_dim)
    sf = torch.tensor(1000.0)
    J_auto = decoder_jacobian(m, assay="a", z_ref=z_ref, size_factor=sf)
    assert J_auto.shape == (in_dim, F)

    # Analytical: rate = exp(z @ W + b) * sf, J[k, f] = sf * exp(z @ W[:,f] + b[f]) * W[k, f]
    W = dec.W.detach()
    b = dec.b.detach()
    log_rate = (z_ref @ W + b).clamp(min=-10, max=8)
    rate = torch.exp(log_rate.clamp(max=15.0)) * sf
    J_analytical = rate.unsqueeze(0) * W  # [in_dim, F]
    diff = (J_auto - J_analytical).abs().max().item()
    assert diff < 1e-4, f"max diff {diff}"


@pytest.mark.integration
def test_decoder_jacobian_dataset_shape(trained_smoke):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]
    J = decoder_jacobian_dataset(model, dm, assay="assay_a", split="all")
    # [N, K_in, F]
    K_in = model.shared_dim + model.private_dims["assay_a"]
    F = dm.assay_shapes["assay_a"]
    assert J.shape[1:] == (K_in, F)
    assert J.shape[0] == len(dm.units)
