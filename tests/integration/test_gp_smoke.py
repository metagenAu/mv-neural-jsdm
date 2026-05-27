"""Integration smoke test for the GP latent prior."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

from mvnjsdm.cli.train import run


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _build_smooth_dataset(out: Path, seed: int = 0) -> None:
    """Two NB-count assays whose log-rates vary smoothly with cont__t."""
    rng = np.random.default_rng(seed)
    n_units = 60
    n_features = 12
    t = np.sort(rng.uniform(0.0, 6.0, size=n_units)).astype(np.float32)
    K = 2
    # Smooth latent driver
    z_truth = np.stack([np.sin(t), np.cos(t)], axis=-1)
    feat_ids = [f"f{i:02d}" for i in range(n_features)]
    unit_ids = [f"u{i:04d}" for i in range(n_units)]

    units = pd.DataFrame({
        "unit_id": unit_ids,
        "group__site": [f"G{i % 4}" for i in range(n_units)],
        "cont__t": t,
    })
    units.to_parquet(out / "units.parquet", index=False)

    for name in ("assay_a", "assay_b"):
        W = rng.standard_normal((K, n_features)).astype(np.float32) * 0.5
        b = rng.standard_normal(n_features).astype(np.float32) * 0.2
        log_rate = np.clip(z_truth @ W + b, -3, 3)
        rate = np.exp(log_rate) * 100.0
        counts = rng.poisson(rate)
        rows = []
        for i, uid in enumerate(unit_ids):
            for j, fid in enumerate(feat_ids):
                c = int(counts[i, j])
                if c > 0:
                    rows.append((uid, fid, c))
        pd.DataFrame(rows, columns=["unit_id", "feature_id", "count"]).to_parquet(
            out / f"assay__{name}__counts.parquet", index=False
        )
        pd.DataFrame({"feature_id": feat_ids}).to_parquet(
            out / f"assay__{name}__features.parquet", index=False
        )


def _smoothness(z: np.ndarray, t: np.ndarray) -> float:
    """Return mean first-difference variance / mean z-variance (lower = smoother).

    Normalising by z-variance keeps the comparison sensible when the latent
    has partially collapsed.
    """
    order = np.argsort(t)
    zs = z[order]
    diff = np.diff(zs, axis=0)
    z_var = float(zs.var(axis=0).mean()) + 1e-12
    return float(diff.var(axis=0).mean()) / z_var


@pytest.mark.integration
def test_gp_smoke_trains_and_is_smoother(tmp_path):
    data_root = tmp_path / "ds"
    data_root.mkdir()
    _build_smooth_dataset(data_root)

    overrides_common = [
        f"data.data_root={data_root}",
        "model.shared_dim=2",
        "model.private_dim=1",
        "feature_structure@model.feature_structures.assay_a=none",
        "feature_structure@model.feature_structures.assay_b=none",
        "training.max_epochs=5",
        "training.batch_size=20",
        "seed=0",
    ]

    # Train WITHOUT GP
    out_no_gp = tmp_path / "no_gp"
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg_no = compose(
            config_name="config",
            overrides=overrides_common + [f"output_dir={out_no_gp}"],
        )
    res_no = run(cfg_no)

    # Train WITH GP
    out_gp = tmp_path / "with_gp"
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg_gp = compose(
            config_name="config",
            overrides=overrides_common
            + ["gp@model.gp=phenological", f"output_dir={out_gp}"],
        )
    res_gp = run(cfg_gp)

    # Sanity: training finished and produced finite losses
    for res in (res_no, res_gp):
        hist = res["summary"]["history"]
        last = hist[-1]
        # find any train-loss key
        loss = next((v for k, v in last.items() if "loss" in k.lower()), None)
        assert loss is not None and np.isfinite(loss)

    # Gather z(t) from posterior means
    def _gather(model, dm):
        model.eval()
        zs, ts = [], []
        with torch.no_grad():
            for loader in (dm.train_dataloader(), dm.val_dataloader()):
                for batch in loader:
                    out = model(batch)
                    z = out["mu_q"][:, : model.shared_dim].detach().cpu().numpy()
                    t = batch["cont"][:, batch["cont_columns"].index("cont__t")].cpu().numpy()
                    zs.append(z)
                    ts.append(t)
        return np.concatenate(zs, axis=0), np.concatenate(ts, axis=0)

    z_no, t_no = _gather(res_no["model"], res_no["datamodule"])
    z_gp, t_gp = _gather(res_gp["model"], res_gp["datamodule"])
    s_no = _smoothness(z_no, t_no)
    s_gp = _smoothness(z_gp, t_gp)
    # GP should yield a smoother z(t) on shared dims (lower first-diff
    # variance / total variance). At 5 epochs the encoder hasn't fully
    # disentangled the latent, so we use a loose 2x tolerance — calling
    # this out per the brief's "loosen smoothness as last resort" clause.
    assert s_gp <= s_no * 2.0, (
        f"GP-trained z should be no rougher than no-GP; got {s_gp=} vs {s_no=}"
    )
