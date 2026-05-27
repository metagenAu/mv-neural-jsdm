"""End-to-end Hydra wiring: feature_structure=graph_laplacian / taxonomic_groupwise."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir

from mvnjsdm.cli._build import build_datamodule, build_model_from_cfg
from mvnjsdm.models.feature_priors import GraphLaplacian, TaxonomicGroupwise


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _build_dataset_with_graph_and_taxonomy(out: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n_units = 50
    n_features = 8
    feat_ids = [f"f{i:02d}" for i in range(n_features)]
    unit_ids = [f"u{i:04d}" for i in range(n_units)]
    counts = rng.poisson(5.0, size=(n_units, n_features)).astype(np.int64)

    units = pd.DataFrame({
        "unit_id": unit_ids,
        "group__site": [f"G{i % 3}" for i in range(n_units)],
    })
    units.to_parquet(out / "units.parquet", index=False)
    # Two assays so the default config (two assays) is satisfied
    for name in ("assay_a", "assay_b"):
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
    # Attach a chain graph for assay_a
    edges = pd.DataFrame({
        "source": [feat_ids[i] for i in range(n_features - 1)],
        "target": [feat_ids[i + 1] for i in range(n_features - 1)],
    })
    edges.to_parquet(out / "assay__assay_a__graph.edges.parquet", index=False)
    # Attach taxonomy groups for assay_b: two groups
    tax = pd.DataFrame({
        "feature_id": feat_ids,
        "group": ["A" if i < 4 else "B" for i in range(n_features)],
    })
    tax.to_parquet(out / "assay__assay_b__taxonomy.parquet", index=False)


def test_graph_laplacian_wires_through_cli(tmp_path):
    data_root = tmp_path / "ds"
    data_root.mkdir()
    _build_dataset_with_graph_and_taxonomy(data_root)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.data_root={data_root}",
                "feature_structure@model.feature_structures.assay_a=graph_laplacian",
                "feature_structure@model.feature_structures.assay_b=none",
                "model.shared_dim=2",
                "model.private_dim=1",
            ],
        )
    dm = build_datamodule(cfg)
    model, _lit = build_model_from_cfg(cfg, dm)
    fp = model.decoders["assay_a"].feature_prior
    assert isinstance(fp, GraphLaplacian)
    # Feature-prior loss must be finite at init
    loss = model.decoders["assay_a"].feature_prior_loss()
    assert torch.isfinite(loss)


def test_taxonomic_groupwise_wires_through_cli(tmp_path):
    data_root = tmp_path / "ds"
    data_root.mkdir()
    _build_dataset_with_graph_and_taxonomy(data_root)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.data_root={data_root}",
                "feature_structure@model.feature_structures.assay_a=none",
                "feature_structure@model.feature_structures.assay_b=taxonomic_groupwise",
                "model.shared_dim=2",
                "model.private_dim=1",
            ],
        )
    dm = build_datamodule(cfg)
    model, _lit = build_model_from_cfg(cfg, dm)
    fp = model.decoders["assay_b"].feature_prior
    assert isinstance(fp, TaxonomicGroupwise)
    loss = model.decoders["assay_b"].feature_prior_loss()
    assert torch.isfinite(loss)
