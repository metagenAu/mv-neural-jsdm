"""Export trained latent representations to a DataFrame for downstream analysis.

The exported frame has one row per unit (or per posterior sample, in long form)
with columns:

    unit_id
    split                            train|val|all
    group__<level>                   propagated from datamodule.units
    env__<name>                      propagated from datamodule.units
    cont__<name>                     propagated from datamodule.units
    latent_shared_0..K_s-1           shared-block posterior means/samples
    latent_private_<assay>_0..K_a-1  per-assay private posterior means/samples

If ``n_samples > 1``, results are in long form with an extra ``sample`` column;
each sample is a reparameterised draw from q(z | x).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch


def _unwrap_model(model: Any) -> Any:
    # accept either MVNeuralJSDM or a LightningModule wrapping it
    if hasattr(model, "model") and hasattr(model.model, "decoders"):
        return model.model
    return model


def _iter_loaders(datamodule: Any, split: str):
    if split == "train":
        yield "train", datamodule.train_dataloader()
    elif split == "val":
        yield "val", datamodule.val_dataloader()
    elif split == "all":
        yield "train", datamodule.train_dataloader()
        yield "val", datamodule.val_dataloader()
    else:
        raise ValueError(f"unknown split: {split}")


def export_latents(
    model: Any,
    datamodule: Any,
    *,
    split: Literal["train", "val", "all"] = "all",
    n_samples: int = 1,
    return_format: Literal["pandas", "mudata"] = "pandas",
) -> pd.DataFrame:
    """Return a DataFrame with metadata + latent columns.

    Parameters
    ----------
    model:
        Either an `MVNeuralJSDM` or a `MVNeuralJSDMLit` wrapping one.
    datamodule:
        The data module used to train (must have been ``setup()``-ed).
    split:
        Which split to export.
    n_samples:
        If 1 (default), uses the posterior mean and returns wide form.
        If >1, draws ``n_samples`` reparameterised samples from
        ``q(z | x)`` for each unit and returns long form with a ``sample``
        column.
    return_format:
        Only "pandas" supported; "mudata" reserved for future work.
    """
    if return_format != "pandas":
        raise NotImplementedError("only pandas return format is supported")

    m = _unwrap_model(model)
    m.eval()

    assay_names = [s.name for s in m.assay_specs]
    shared = m.shared_dim
    private_dims = m.private_dims
    private_offsets = m.private_offsets

    units = datamodule.units.set_index("unit_id", drop=False)
    meta_cols = [
        c
        for c in units.columns
        if c == "unit_id"
        or c.startswith("group__")
        or c.startswith("env__")
        or c.startswith("cont__")
    ]

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for loader_name, loader in _iter_loaders(datamodule, split):
            for batch in loader:
                out = m(batch)
                mu = out["mu_q"].detach().cpu().numpy()
                lv = out["logvar_q"].detach().cpu().numpy()
                uids = batch["unit_id"]
                B = mu.shape[0]
                if n_samples == 1:
                    Z_list = [(0, mu)]
                else:
                    std = np.exp(0.5 * lv)
                    Z_list = []
                    for s_idx in range(n_samples):
                        eps = np.random.standard_normal(mu.shape).astype(mu.dtype)
                        Z_list.append((s_idx, mu + std * eps))
                for s_idx, Z in Z_list:
                    for i in range(B):
                        uid = uids[i]
                        row: dict[str, Any] = {"unit_id": uid, "split": loader_name}
                        if n_samples > 1:
                            row["sample"] = int(s_idx)
                        # metadata
                        if uid in units.index:
                            urow = units.loc[uid]
                            for c in meta_cols:
                                if c == "unit_id":
                                    continue
                                row[c] = urow[c]
                        # latent columns
                        for k in range(shared):
                            row[f"latent_shared_{k}"] = float(Z[i, k])
                        for name in assay_names:
                            off = private_offsets[name]
                            d = private_dims[name]
                            for k in range(d):
                                row[f"latent_private_{name}_{k}"] = float(Z[i, off + k])
                        rows.append(row)

    df = pd.DataFrame(rows)
    if n_samples == 1 and not df.empty:
        # de-duplicate (a unit may appear in both train and val readers if val
        # loader falls back to train when val is empty)
        df = df.drop_duplicates(subset=["unit_id"]).reset_index(drop=True)
    return df


def write_latents_parquet(df: pd.DataFrame, path: str | Path) -> None:
    df.to_parquet(Path(path), index=False)


def write_latents_csv(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(Path(path), index=False)
