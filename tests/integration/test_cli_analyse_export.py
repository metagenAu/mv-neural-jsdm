"""Smoke-test the analyse + export CLIs by piggy-backing on ``trained_smoke``.

Drives ``cli.analyse.run`` and ``cli.export.run`` programmatically against the
shared ``trained_smoke`` fixture's run directory.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf


@pytest.mark.integration
def test_analyse_and_export_cli(trained_smoke):
    run_dir = Path(trained_smoke["out_dir"])
    # check the trainer wrote the artefacts the CLIs need
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "model.ckpt").exists()

    # --- analyse
    from mvnjsdm.cli.analyse import run as analyse_run

    cfg = OmegaConf.create(
        {
            "run_dir": str(run_dir),
            "analyses": [
                "variance_partition",
                "latent_interaction",
                "cross_assay",
                "within_assay",
                "ordination",
            ],
            "levels": None,
            "env_covariates": None,
            "n_components": 2,
        }
    )
    out = analyse_run(cfg)
    analysis_dir = Path(out["out_dir"])
    assert (analysis_dir / "variance_partition.csv").exists()
    assert (analysis_dir / "latent_interaction.csv").exists()
    assert (analysis_dir / "ordination.csv").exists()
    # one cross-assay file for the assay_a x assay_b pair
    assert (analysis_dir / "cross_assay_assay_a__assay_b.parquet").exists()
    # within-assay files for each assay
    assert (analysis_dir / "within_assay_assay_a.parquet").exists()
    assert (analysis_dir / "within_assay_assay_b.parquet").exists()

    # sanity-check shapes
    cross = pd.read_parquet(analysis_dir / "cross_assay_assay_a__assay_b.parquet")
    assert cross.shape[0] > 0 and cross.shape[1] > 0

    # --- export
    from mvnjsdm.cli.export import run as export_run

    cfg = OmegaConf.create(
        {
            "run_dir": str(run_dir),
            "split": "all",
            "n_samples": 1,
            "formats": ["parquet", "csv"],
            "sem_export": True,
            "sem_filename": "latents_sem.csv",
        }
    )
    out = export_run(cfg)
    assert (run_dir / "latents.parquet").exists()
    assert (run_dir / "latents.csv").exists()
    assert (run_dir / "latents_sem.csv").exists()
    assert out["rows"] > 0


@pytest.mark.integration
def test_analyse_missing_run_dir_errors(tmp_path):
    from mvnjsdm.cli.analyse import run as analyse_run

    cfg = OmegaConf.create({"run_dir": str(tmp_path / "does_not_exist"), "analyses": []})
    with pytest.raises(FileNotFoundError):
        analyse_run(cfg)
