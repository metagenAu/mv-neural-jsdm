"""Mixed-assay end-to-end integration test.

Trains an MVNeuralJSDM on three synthetic assays (NB counts with a phylo prior,
Gaussian continuous, Bernoulli binary) and exercises latent export, cross-assay
interaction, and per-assay decoder Jacobians.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from mvnjsdm.analysis.interaction_matrix import cross_assay_interaction_matrix
from mvnjsdm.analysis.latent_export import export_latents
from mvnjsdm.cli.train import run as train_run
from mvnjsdm.interpret.decoder_jacobian import decoder_jacobian
from tests.conftest import make_mixed_synthetic


def _build_cfg(data_root: str, out_dir: str) -> "OmegaConf":
    cfg = {
        "seed": 0,
        "output_dir": out_dir,
        "data": {"data_root": data_root},
        "training": {
            "max_epochs": 5,
            "batch_size": 32,
            "lr": 5e-3,
            "val_frac": 0.15,
            "log_every_n_steps": 1,
            "progress_bar": False,
        },
        "model": {
            "shared_dim": 4,
            "private_dim": 2,
            "fusion": "mopoe",
            "beta_shared": 1.0,
            "beta_private": 1.0,
            "assays": {
                "assay_counts": {
                    "kind": "counts",
                    "likelihood": "nb",
                    "size_factor": True,
                    "transform": "identity",
                    "encoder_config": {},
                    "decoder_config": {},
                },
                "assay_chem": {
                    "kind": "continuous",
                    "likelihood": "gaussian_masked",
                    "size_factor": False,
                    "transform": "identity",
                    "encoder_config": {},
                    "decoder_config": {},
                },
                "assay_binary": {
                    "kind": "binary",
                    "likelihood": "bernoulli",
                    "size_factor": False,
                    "transform": "identity",
                    "encoder_config": {},
                    "decoder_config": {},
                },
            },
            "feature_structures": {
                "assay_counts": {"kind": "pagel", "init_lambda": 0.5},
                "assay_chem": {"kind": "none"},
                "assay_binary": {"kind": "none"},
            },
            "env": {"enabled": False},
            "hierarchy": {"levels": []},
            "ar": {"enabled": False},
        },
    }
    return OmegaConf.create(cfg)


@pytest.mark.integration
def test_mixed_assay_end_to_end(tmp_path):
    data_dir, truth = make_mixed_synthetic(tmp_path / "data", seed=0)
    out_dir = tmp_path / "run"

    cfg = _build_cfg(str(data_dir), str(out_dir))
    result = train_run(cfg)
    summary = result["summary"]
    history = summary["history"]
    assert len(history) >= 3

    # All three reconstruction losses are logged and finite.
    recon_keys = {f"train/recon_{a}" for a in ("assay_counts", "assay_chem", "assay_binary")}
    present = set(history[-1].keys())
    missing = recon_keys - present
    assert not missing, f"missing recon log keys: {missing}"
    for row in history:
        for k, v in row.items():
            if k == "epoch":
                continue
            assert v == v, f"NaN in {k} at epoch {row.get('epoch')}"

    # Loss decreased over last 3 epochs (allow 5% noise).
    losses = [r.get("train/loss") for r in history if "train/loss" in r]
    if len(losses) >= 3:
        last3 = losses[-3:]
        assert last3[-1] <= last3[0] * 1.05, f"loss did not decrease: {last3}"

    model = result["model"]
    dm = result["datamodule"]
    lit = result["lit"]

    # --- latent export
    df = export_latents(lit, dm, split="all", n_samples=1)
    assert df["unit_id"].nunique() == truth["n_units"]
    shared_cols = [c for c in df.columns if c.startswith("latent_shared_")]
    assert len(shared_cols) == 4

    # --- cross-assay interaction matrix (counts x chem)
    cross = cross_assay_interaction_matrix(
        lit, dm, assay_a="assay_counts", assay_b="assay_chem"
    )
    assert cross.shape == (truth["F_counts"], truth["F_chem"])

    # --- decoder jacobian shapes for each assay
    import torch

    z_zero = torch.zeros(model.shared_dim + model.private_dims["assay_counts"])
    J_c = decoder_jacobian(
        model,
        assay="assay_counts",
        z_ref=z_zero,
        size_factor=torch.tensor(1000.0),
    )
    assert J_c.shape == (model.shared_dim + model.private_dims["assay_counts"], truth["F_counts"])

    J_g = decoder_jacobian(
        model,
        assay="assay_chem",
        z_ref=torch.zeros(model.shared_dim + model.private_dims["assay_chem"]),
    )
    assert J_g.shape == (model.shared_dim + model.private_dims["assay_chem"], truth["F_chem"])

    J_b = decoder_jacobian(
        model,
        assay="assay_binary",
        z_ref=torch.zeros(model.shared_dim + model.private_dims["assay_binary"]),
    )
    assert J_b.shape == (model.shared_dim + model.private_dims["assay_binary"], truth["F_binary"])
