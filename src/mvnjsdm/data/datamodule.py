"""Lightning DataModule wrapping a MuData object.

Each batch is a dict of:

    {
        "unit_id": list[str],
        "assays": {name: {"x": Tensor[B,F], "mask": Tensor[B,F], "sf": Tensor[B] | None}},
        "group_ids": {level: LongTensor[B]},
        "env": Tensor[B,E] | None,
        "cont": Tensor[B,C] | None,
        "batch_cov": Tensor[B,K] | None,
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mudata as md
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from .continuous_index import ContinuousIndexScaler
from .loaders import load_dataset, units_frame
from .schema import batch_cols, cont_cols, env_cols, group_cols
from .splits import by_group_split
from .transforms import library_size


class _MVDataset(Dataset):
    def __init__(
        self,
        mud: md.MuData,
        units: pd.DataFrame,
        idx: np.ndarray,
        group_levels: list[str],
        group_codes: dict[str, np.ndarray],
        env_cols_: list[str],
        cont_cols_: list[str],
        batch_cols_: list[str],
        assay_size_factor: dict[str, bool],
    ) -> None:
        self.mud = mud
        self.units = units
        self.idx = idx
        self.group_levels = group_levels
        self.group_codes = group_codes
        self.env_cols = env_cols_
        self.cont_cols = cont_cols_
        self.batch_cols = batch_cols_
        self.assay_size_factor = assay_size_factor

        # Pre-stack per-assay matrices
        self._X: dict[str, np.ndarray] = {}
        self._mask: dict[str, np.ndarray] = {}
        for name, ad_obj in mud.mod.items():
            X = np.asarray(ad_obj.X, dtype=np.float32)
            mask = np.ones_like(X, dtype=np.float32)
            # treat NaN as masked
            nan_mask = ~np.isnan(X)
            mask = mask * nan_mask.astype(np.float32)
            X = np.where(nan_mask, X, 0.0)
            self._X[name] = X
            self._mask[name] = mask

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = int(self.idx[i])
        out: dict[str, Any] = {
            "row": row,
            "unit_id": str(self.units.iloc[row]["unit_id"]),
        }
        out["assays"] = {}
        for name in self.mud.mod.keys():
            x = self._X[name][row]
            m = self._mask[name][row]
            d = {"x": torch.from_numpy(x), "mask": torch.from_numpy(m)}
            if self.assay_size_factor.get(name, False):
                d["sf"] = torch.tensor(float(x.sum()) + 1.0, dtype=torch.float32)
            out["assays"][name] = d

        out["group_ids"] = {
            lev: torch.tensor(int(self.group_codes[lev][row]), dtype=torch.long)
            for lev in self.group_levels
        }
        if self.env_cols:
            out["env"] = torch.tensor(
                self.units.iloc[row][self.env_cols].to_numpy(dtype=np.float32)
            )
        if self.cont_cols:
            out["cont"] = torch.tensor(
                self.units.iloc[row][self.cont_cols].to_numpy(dtype=np.float32)
            )
            out["cont_columns"] = list(self.cont_cols)
        if self.batch_cols:
            out["batch_cov"] = torch.tensor(
                self.units.iloc[row][self.batch_cols].to_numpy(dtype=np.float32)
            )
        return out


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "unit_id": [b["unit_id"] for b in batch],
        "assays": {},
        "group_ids": {},
    }
    assay_names = list(batch[0]["assays"].keys())
    for name in assay_names:
        out["assays"][name] = {
            "x": torch.stack([b["assays"][name]["x"] for b in batch]),
            "mask": torch.stack([b["assays"][name]["mask"] for b in batch]),
        }
        if "sf" in batch[0]["assays"][name]:
            out["assays"][name]["sf"] = torch.stack(
                [b["assays"][name]["sf"] for b in batch]
            )
        else:
            out["assays"][name]["sf"] = None
    for lev in batch[0]["group_ids"]:
        out["group_ids"][lev] = torch.stack([b["group_ids"][lev] for b in batch])
    for key in ("env", "cont", "batch_cov"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = None
    if "cont_columns" in batch[0]:
        out["cont_columns"] = batch[0]["cont_columns"]
        cols = out["cont_columns"]
        if out.get("cont") is not None and cols:
            out["continuous_indices"] = {
                c: out["cont"][:, i] for i, c in enumerate(cols)
            }
    return out


class MVNeuralJSDMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: str | Path,
        batch_size: int = 32,
        val_frac: float = 0.15,
        seed: int = 0,
        assay_size_factor: dict[str, bool] | None = None,
        split_group_col: str | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.seed = seed
        self.assay_size_factor = assay_size_factor or {}
        self.split_group_col = split_group_col
        self.num_workers = num_workers

        self.mud: md.MuData | None = None
        self.units: pd.DataFrame | None = None
        self.group_levels: list[str] = []
        self.group_codes: dict[str, np.ndarray] = {}
        self.group_cardinality: dict[str, int] = {}
        self.env_cols: list[str] = []
        self.cont_cols: list[str] = []
        self.batch_cols: list[str] = []
        self._splits: dict[str, np.ndarray] | None = None
        self._cont_scaler = ContinuousIndexScaler()

    # -- properties -----
    @property
    def assay_shapes(self) -> dict[str, int]:
        if self.mud is None:
            self.setup()
        return {name: int(self.mud[name].X.shape[1]) for name in self.mud.mod.keys()}

    def get_tree_C(self, assay_name: str) -> np.ndarray | None:
        if self.mud is None:
            self.setup()
        ad_obj = self.mud[assay_name]
        if "tree_C" in ad_obj.varm:
            return np.asarray(ad_obj.varm["tree_C"])
        return None

    def get_graph_L(self, assay_name: str) -> np.ndarray | None:
        if self.mud is None:
            self.setup()
        ad_obj = self.mud[assay_name]
        if "graph_L" in ad_obj.varm:
            return np.asarray(ad_obj.varm["graph_L"])
        return None

    def get_taxonomy_groups(self, assay_name: str) -> np.ndarray | None:
        if self.mud is None:
            self.setup()
        ad_obj = self.mud[assay_name]
        if "taxonomy_groups" in ad_obj.varm:
            return np.asarray(ad_obj.varm["taxonomy_groups"]).astype(np.int64).ravel()
        return None

    # -- lifecycle ------
    def prepare_data(self) -> None:  # noqa: D401
        pass

    def setup(self, stage: str | None = None) -> None:
        if self.mud is not None:
            return
        self.mud = load_dataset(self.data_root)
        self.units = units_frame(self.mud)
        self.group_levels = group_cols(self.units)
        self.env_cols = env_cols(self.units)
        self.cont_cols = cont_cols(self.units)
        self.batch_cols = batch_cols(self.units)

        # cont scaling
        if self.cont_cols:
            self.units = self._cont_scaler.fit_transform(self.units)

        # categorical encoding for groups & batch
        for lev in self.group_levels:
            cats = self.units[lev].astype(str)
            uniq = sorted(cats.unique().tolist())
            mp = {v: i for i, v in enumerate(uniq)}
            self.group_codes[lev] = cats.map(mp).to_numpy(dtype=np.int64)
            self.group_cardinality[lev] = len(uniq)

        # batch-cov as one-hot per col is left to the model; we pass raw codes
        # (here we keep them as float-encoded ordinals for simplicity).
        for bc in list(self.batch_cols):
            cats = self.units[bc].astype(str)
            uniq = sorted(cats.unique().tolist())
            mp = {v: float(i) for i, v in enumerate(uniq)}
            self.units[bc] = cats.map(mp).astype(float)

        split_col = self.split_group_col
        if split_col is None and self.group_levels:
            split_col = self.group_levels[-1]
        if split_col is None:
            split_col = "__none__"
        self._splits = by_group_split(
            self.units, split_col, val_frac=self.val_frac, seed=self.seed
        )

    # -- dataloaders ---
    def _make_loader(self, split: str, shuffle: bool) -> DataLoader:
        assert self.mud is not None
        ds = _MVDataset(
            self.mud,
            self.units,
            self._splits[split],
            self.group_levels,
            self.group_codes,
            self.env_cols,
            self.cont_cols,
            self.batch_cols,
            self.assay_size_factor,
        )
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_collate,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if len(self._splits["val"]) == 0:
            return self._make_loader("train", shuffle=False)
        return self._make_loader("val", shuffle=False)
