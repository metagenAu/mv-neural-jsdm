from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mvnjsdm.analysis.latent_export import (
    export_latents,
    write_latents_csv,
    write_latents_parquet,
)


@pytest.mark.integration
def test_export_latents_wide(trained_smoke, tmp_path):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]

    df = export_latents(model, dm, split="all", n_samples=1)
    assert "unit_id" in df.columns
    n_units = len(dm.units)
    assert df.shape[0] == n_units
    assert df["unit_id"].is_unique

    # latent columns present
    lat_cols = [c for c in df.columns if c.startswith("latent_")]
    assert any(c.startswith("latent_shared_") for c in lat_cols)
    assert any(c.startswith("latent_private_") for c in lat_cols)

    # group metadata propagated
    assert any(c.startswith("group__") for c in df.columns)


@pytest.mark.integration
def test_export_latents_samples_long(trained_smoke):
    df = export_latents(
        trained_smoke["model"], trained_smoke["datamodule"], split="all", n_samples=3
    )
    n_units = len(trained_smoke["datamodule"].units)
    assert "sample" in df.columns
    assert df.shape[0] == 3 * n_units


@pytest.mark.integration
def test_write_latents_parquet_roundtrip(trained_smoke, tmp_path: Path):
    df = export_latents(trained_smoke["model"], trained_smoke["datamodule"], split="all")
    p = tmp_path / "lat.parquet"
    write_latents_parquet(df, p)
    df2 = pd.read_parquet(p)
    assert set(df2.columns) == set(df.columns)
    assert df2.shape == df.shape


@pytest.mark.integration
def test_write_latents_csv(trained_smoke, tmp_path: Path):
    df = export_latents(trained_smoke["model"], trained_smoke["datamodule"], split="all")
    p = tmp_path / "lat.csv"
    write_latents_csv(df, p)
    assert p.exists()
    df2 = pd.read_csv(p)
    assert df2.shape == df.shape
