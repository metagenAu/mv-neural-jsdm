from __future__ import annotations

import torch

from mvnjsdm.models.mvnjsdm import AssaySpec, MVNeuralJSDM
from mvnjsdm.training.losses import LossWeights, compute_loss


def _make_batch(B: int, F_a: int, F_b: int, K: int) -> dict:
    return {
        "assays": {
            "assay_a": {
                "x": torch.randint(0, 5, (B, F_a)).float(),
                "mask": torch.ones(B, F_a),
                "sf": torch.full((B,), 100.0),
            },
            "assay_b": {
                "x": torch.randint(0, 5, (B, F_b)).float(),
                "mask": torch.ones(B, F_b),
                "sf": torch.full((B,), 100.0),
            },
        },
        "group_ids": {},
        "env": None,
    }


def test_loss_finite_and_kl_nonneg():
    F_a, F_b = 10, 12
    specs = [
        AssaySpec(name="assay_a", kind="counts", likelihood="nb", n_features=F_a, private_dim=2, size_factor=True),
        AssaySpec(name="assay_b", kind="counts", likelihood="nb", n_features=F_b, private_dim=2, size_factor=True),
    ]
    model = MVNeuralJSDM(assay_specs=specs, shared_dim=4, fusion="poe")
    batch = _make_batch(B=8, F_a=F_a, F_b=F_b, K=4 + 2 + 2)
    out = model(batch)
    parts = compute_loss(model, batch, out, LossWeights())
    assert torch.isfinite(parts["loss"])
    assert parts["kl_shared"] >= 0
    for name in ("assay_a", "assay_b"):
        assert parts[f"kl_priv_{name}"] >= 0
