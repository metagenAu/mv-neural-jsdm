"""Hydra-driven post-hoc analysis CLI.

Usage::

    python -m mvnjsdm.cli.analyse run_dir=outputs/2026-05-27_... \
        analyses=[variance_partition,latent_interaction,cross_assay,within_assay,ordination]

Reads ``run_dir/config.yaml`` + ``run_dir/model.ckpt``, reconstructs the
LightningModule and datamodule, and writes per-analysis files under
``run_dir/analysis/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from ..analysis.interaction_matrix import (
    cross_assay_interaction_matrix,
    latent_interaction_matrix,
    within_assay_interaction_matrix,
)
from ..analysis.latent_export import export_latents
from ..analysis.ordination import latent_pca
from ..analysis.variance_partition import variance_partition
from ._reload import reload_from_run_dir

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


def run(cfg: DictConfig) -> dict[str, Any]:
    if cfg.get("run_dir", None) in (None, "???"):
        raise ValueError("run_dir is required")
    run_dir = Path(cfg.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    lit, dm, _train_cfg = reload_from_run_dir(run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyses = list(cfg.get("analyses", []) or [])

    def _to_list(v: Any) -> list | None:
        if v is None:
            return None
        if OmegaConf.is_config(v):
            return OmegaConf.to_container(v, resolve=True)
        return list(v)

    levels = _to_list(cfg.get("levels", None))
    env_covariates = _to_list(cfg.get("env_covariates", None))
    n_components = int(cfg.get("n_components", 2))

    written: list[Path] = []
    assay_names = list(dm.assay_shapes.keys())

    if "variance_partition" in analyses:
        df = variance_partition(
            lit, dm, levels=levels, env_covariates=env_covariates
        )
        p = out_dir / "variance_partition.csv"
        df.to_csv(p)
        written.append(p)

    if "latent_interaction" in analyses:
        cov = latent_interaction_matrix(lit, dm)
        p = out_dir / "latent_interaction.csv"
        import numpy as np
        import pandas as pd

        cols = [f"z{i}" for i in range(cov.shape[0])]
        pd.DataFrame(np.asarray(cov), index=cols, columns=cols).to_csv(p)
        written.append(p)

    if "cross_assay" in analyses:
        for i, a in enumerate(assay_names):
            for b in assay_names[i + 1 :]:
                df = cross_assay_interaction_matrix(lit, dm, assay_a=a, assay_b=b)
                p = out_dir / f"cross_assay_{a}__{b}.parquet"
                df.to_parquet(p)
                written.append(p)

    if "within_assay" in analyses:
        for a in assay_names:
            df = within_assay_interaction_matrix(lit, dm, assay=a)
            p = out_dir / f"within_assay_{a}.parquet"
            df.to_parquet(p)
            written.append(p)

    if "ordination" in analyses:
        latents = export_latents(lit, dm, split="all", n_samples=1)
        pca_df = latent_pca(latents, n_components=n_components)
        p = out_dir / "ordination.csv"
        pca_df.to_csv(p, index=False)
        written.append(p)

    print(f"[analyse] wrote {len(written)} files under {out_dir}:")
    for p in written:
        print(f"  - {p.name}")
    return {"written": [str(p) for p in written], "out_dir": str(out_dir)}


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="analyse")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
