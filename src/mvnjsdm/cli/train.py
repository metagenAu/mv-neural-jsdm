"""Hydra-driven training entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.datamodule import MVNeuralJSDMDataModule
from ..models.hierarchy import LevelSpec
from ..models.mvnjsdm import AssaySpec, MVNeuralJSDM
from ..training.callbacks import LatentCollapseMonitor
from ..training.lightning_module import MVNeuralJSDMLit
from ..training.losses import LossWeights

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


def _build_assay_specs(cfg: DictConfig, dm: MVNeuralJSDMDataModule) -> list[AssaySpec]:
    specs: list[AssaySpec] = []
    private_dim_default = int(cfg.model.private_dim)
    shapes = dm.assay_shapes
    fs_cfgs = OmegaConf.to_container(cfg.model.get("feature_structures", {}), resolve=True) or {}
    for name, acfg in cfg.model.assays.items():
        d = OmegaConf.to_container(acfg, resolve=True)
        n_features = shapes[name]
        fs = fs_cfgs.get(name, {"kind": "none"})
        feature_structure = fs.get("kind", "none")
        init_lambda = float(fs.get("init_lambda", 0.5))
        tree_C = dm.get_tree_C(name) if feature_structure in ("pagel", "brownian") else None
        specs.append(
            AssaySpec(
                name=name,
                kind=d["kind"],
                likelihood=d["likelihood"],
                n_features=n_features,
                private_dim=int(d.get("private_dim", private_dim_default)),
                size_factor=bool(d.get("size_factor", False)),
                tree_C=tree_C,
                feature_structure=feature_structure,
                init_lambda=init_lambda,
            )
        )
    return specs


def _build_hierarchy_levels(cfg: DictConfig, dm: MVNeuralJSDMDataModule) -> list[LevelSpec]:
    h = cfg.model.get("hierarchy", None)
    if h is None:
        return []
    levels_cfg = OmegaConf.to_container(h.get("levels", []), resolve=True) or []
    specs: list[LevelSpec] = []
    for entry in levels_cfg:
        name = entry["name"]
        mode = entry.get("mode", "hierarchical_prior")
        if name not in dm.group_cardinality:
            # silently skip levels not present in data
            continue
        specs.append(LevelSpec(name=name, n_groups=dm.group_cardinality[name], mode=mode))
    return specs


def _export_latents(model: MVNeuralJSDM, dm: MVNeuralJSDMDataModule, out_dir: Path) -> Path:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for loader_name, loader in [("train", dm.train_dataloader()), ("val", dm.val_dataloader())]:
            for batch in loader:
                out = model(batch)
                mu = out["mu_q"].cpu().numpy()
                for i, uid in enumerate(batch["unit_id"]):
                    row = {"unit_id": uid, "split": loader_name}
                    for k in range(mu.shape[1]):
                        row[f"z{k}"] = float(mu[i, k])
                    rows.append(row)
    df = pd.DataFrame(rows).drop_duplicates(subset=["unit_id"])
    p = out_dir / "latents.parquet"
    df.to_parquet(p, index=False)
    return p


def run(cfg: DictConfig) -> dict[str, Any]:
    """Programmatic entrypoint, callable from tests."""
    seed = int(cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- data
    data_root = cfg.data.data_root
    bs = int(cfg.training.get("batch_size", 32))
    size_factor_map: dict[str, bool] = {}
    for name, acfg in cfg.model.assays.items():
        size_factor_map[name] = bool(acfg.get("size_factor", False))
    dm = MVNeuralJSDMDataModule(
        data_root=data_root,
        batch_size=bs,
        val_frac=float(cfg.training.get("val_frac", 0.15)),
        seed=seed,
        assay_size_factor=size_factor_map,
    )
    dm.setup()

    # --- model
    assay_specs = _build_assay_specs(cfg, dm)
    hierarchy_levels = _build_hierarchy_levels(cfg, dm)

    env_dim = len(dm.env_cols) if cfg.model.get("env", {}).get("enabled", False) else 0
    model = MVNeuralJSDM(
        assay_specs=assay_specs,
        shared_dim=int(cfg.model.shared_dim),
        fusion=str(cfg.model.fusion),
        hierarchy_levels=hierarchy_levels,
        env_dim=env_dim,
        ar_enabled=bool(cfg.model.get("ar", {}).get("enabled", False)),
    )

    weights = LossWeights(
        beta_shared=float(cfg.model.beta_shared),
        beta_private={name: float(cfg.model.beta_private) for name in dm.assay_shapes},
    )
    lit = MVNeuralJSDMLit(model=model, weights=weights, lr=float(cfg.training.get("lr", 1e-3)))

    trainer = pl.Trainer(
        max_epochs=int(cfg.training.max_epochs),
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=bool(cfg.training.get("progress_bar", False)),
        log_every_n_steps=int(cfg.training.get("log_every_n_steps", 1)),
        callbacks=[LatentCollapseMonitor()],
        default_root_dir=str(out_dir),
        deterministic=False,
    )

    # capture training/validation losses
    history: list[dict[str, float]] = []

    class _Hist(pl.Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            row = {"epoch": int(trainer.current_epoch)}
            for k, v in trainer.callback_metrics.items():
                try:
                    row[str(k)] = float(v)
                except Exception:
                    continue
            history.append(row)

    trainer.callbacks.append(_Hist())
    trainer.fit(lit, datamodule=dm)

    # export
    latents_path = _export_latents(model, dm, out_dir)

    # summary
    feature_priors_info = {}
    for name in dm.assay_shapes:
        fp = model.decoders[name].feature_prior
        info: dict[str, Any] = {"kind": fp.__class__.__name__}
        if hasattr(fp, "lambda_value"):
            info["lambda"] = float(fp.lambda_value().detach().cpu())
        if hasattr(fp, "sigma2"):
            info["sigma2"] = float(fp.sigma2().detach().cpu())
        feature_priors_info[name] = info

    hierarchy_info: dict[str, Any] = {}
    for lv_name, lv in model.hierarchy.levels.items():
        if hasattr(lv, "raw_logvar") and lv.raw_logvar is not None:
            hierarchy_info[lv_name] = {
                "logvar_mean": float(lv.raw_logvar.detach().cpu().mean()),
                "emb_var": float(lv.emb.weight.detach().cpu().var().item()),
            }

    summary = {
        "history": history,
        "latents_path": str(latents_path),
        "feature_priors": feature_priors_info,
        "hierarchy": hierarchy_info,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"model": model, "datamodule": dm, "summary": summary, "out_dir": out_dir}


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
