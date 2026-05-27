from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from mvnjsdm.cli.train import run

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.mark.integration
def test_smoke_end_to_end(tmp_path, synthetic_smoke_dir, synthetic_smoke_truth, capsys):
    # Resolve data path absolute to repo
    data_root = str(synthetic_smoke_dir)
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "+experiment=smoke",
                f"data.data_root={data_root}",
                f"output_dir={out_dir}",
                "seed=0",
            ],
        )
    result = run(cfg)
    elapsed = time.time() - start
    print(f"\n[smoke] elapsed wall clock: {elapsed:.1f}s")

    summary = result["summary"]
    history = summary["history"]
    assert len(history) >= 3

    # No NaN
    for row in history:
        for k, v in row.items():
            if k == "epoch":
                continue
            assert v == v, f"NaN in {k} at epoch {row.get('epoch')}"

    # Monotonic-ish decrease over last 3 epochs (allow 5% noise)
    losses = [row.get("train/loss") for row in history if "train/loss" in row]
    if len(losses) >= 3:
        last3 = losses[-3:]
        assert last3[-1] <= last3[0] * 1.05, f"loss did not decrease in last 3 epochs: {last3}"

    # Latents export exists
    latents_path = Path(summary["latents_path"])
    assert latents_path.exists()
    df = pd.read_parquet(latents_path)
    assert "unit_id" in df.columns
    assert df.shape[0] > 0

    # Recovered Pagel lambda within 0.2 (loose)
    fps = summary["feature_priors"]
    for name in ("assay_a", "assay_b"):
        info = fps[name]
        if "lambda" in info:
            lam_true = synthetic_smoke_truth["lambda_true"][name]
            assert abs(info["lambda"] - lam_true) < 0.5, (
                f"{name}: recovered lambda={info['lambda']} vs true={lam_true}"
            )

    assert elapsed < 180, f"smoke test took {elapsed:.1f}s (>180s budget)"
