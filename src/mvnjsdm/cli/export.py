"""Hydra-driven latent-export CLI.

Loads a trained run and writes ``latents.parquet`` (and optionally CSV /
SEM-ready CSV) under ``run_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from ..analysis.latent_export import (
    export_latents,
    write_latents_csv,
    write_latents_parquet,
)
from ..analysis.sem_export import to_sem_csv
from ._reload import reload_from_run_dir

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


def run(cfg: DictConfig) -> dict[str, Any]:
    if cfg.get("run_dir", None) in (None, "???"):
        raise ValueError("run_dir is required")
    run_dir = Path(cfg.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    lit, dm, _train_cfg = reload_from_run_dir(run_dir)
    split = str(cfg.get("split", "all"))
    n_samples = int(cfg.get("n_samples", 1))
    formats = list(cfg.get("formats", ["parquet"]) or [])

    df = export_latents(lit, dm, split=split, n_samples=n_samples)
    written: list[Path] = []
    if "parquet" in formats:
        p = run_dir / "latents.parquet"
        write_latents_parquet(df, p)
        written.append(p)
    if "csv" in formats:
        p = run_dir / "latents.csv"
        write_latents_csv(df, p)
        written.append(p)

    if bool(cfg.get("sem_export", False)):
        sem_path = run_dir / str(cfg.get("sem_filename", "latents_sem.csv"))
        to_sem_csv(df, sem_path)
        written.append(sem_path)

    print(f"[export] wrote {len(written)} files under {run_dir}:")
    for p in written:
        print(f"  - {p.name}  ({p.stat().st_size} bytes)")
    return {"written": [str(p) for p in written], "rows": len(df)}


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="export")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
