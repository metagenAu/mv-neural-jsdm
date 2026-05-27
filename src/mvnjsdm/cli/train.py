"""Hydra-driven training entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from ..data.datamodule import MVNeuralJSDMDataModule
from ..models.mvnjsdm import MVNeuralJSDM
from ..training.callbacks import LatentCollapseMonitor
from ._build import build_datamodule, build_model_from_cfg

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")


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
    dm = build_datamodule(cfg)

    # --- model
    model, lit = build_model_from_cfg(cfg, dm)

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

    # export latents (simple wide form)
    latents_path = _export_latents(model, dm, out_dir)

    # persist resolved config + state_dict so analyse/export CLIs can reload
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    torch.save({"state_dict": lit.state_dict()}, out_dir / "model.ckpt")

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
    return {"model": model, "datamodule": dm, "lit": lit, "summary": summary, "out_dir": out_dir}


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
