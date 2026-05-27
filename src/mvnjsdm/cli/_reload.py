"""Reconstruct a trained Lightning model + datamodule from a run directory."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from ..data.datamodule import MVNeuralJSDMDataModule
from ..training.lightning_module import MVNeuralJSDMLit
from ._build import build_datamodule, build_model_from_cfg


def reload_from_run_dir(
    run_dir: str | Path,
) -> tuple[MVNeuralJSDMLit, MVNeuralJSDMDataModule, DictConfig]:
    """Load resolved config + checkpoint from ``run_dir`` and rebuild model.

    Expects ``run_dir/config.yaml`` and ``run_dir/model.ckpt`` (the latter as
    saved by :func:`cli.train.run`).
    """
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    ckpt_path = run_dir / "model.ckpt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing config.yaml under {run_dir}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing model.ckpt under {run_dir}")

    cfg = OmegaConf.load(cfg_path)
    dm = build_datamodule(cfg)
    _model, lit = build_model_from_cfg(cfg, dm)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    lit.load_state_dict(state)
    lit.eval()
    return lit, dm, cfg
