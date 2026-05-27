"""Disk -> MuData loader for the mv-neural-jsdm dataset convention."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import mudata as md
import numpy as np
import pandas as pd

from .feature_structures import graph_laplacian, tree_variance_covariance
from .schema import DatasetManifest, validate_units_frame


def _counts_long_to_wide(
    df_long: pd.DataFrame,
    unit_index: pd.Index,
    feature_ids: list[str] | None,
) -> tuple[np.ndarray, list[str], list[str]]:
    if feature_ids is None:
        feature_ids = sorted(df_long["feature_id"].unique().tolist())
    feat_idx = {f: i for i, f in enumerate(feature_ids)}
    unit_idx = {u: i for i, u in enumerate(unit_index)}
    M = np.zeros((len(unit_index), len(feature_ids)), dtype=np.float32)
    for u, f, c in zip(
        df_long["unit_id"].to_numpy(),
        df_long["feature_id"].to_numpy(),
        df_long["count"].to_numpy(),
    ):
        ui = unit_idx.get(u)
        fi = feat_idx.get(f)
        if ui is None or fi is None:
            continue
        M[ui, fi] = float(c)
    return M, list(unit_index), feature_ids


def load_dataset(root: str | Path) -> md.MuData:
    """Load a dataset directory into a MuData object.

    Each assay becomes an AnnData modality. ``obs`` mirrors the unit manifest.
    Feature structure (tree covariance, graph Laplacian) is stored in
    ``adata.varm`` / ``adata.uns`` per modality.
    """
    manifest = DatasetManifest.discover(root)
    units = pd.read_parquet(manifest.units_path)
    validate_units_frame(units)
    units = units.set_index("unit_id", drop=False)
    units.index.name = "unit_id"

    modalities: dict[str, ad.AnnData] = {}

    for assay in manifest.assays:
        feature_ids: list[str] | None = None
        if assay.features_path is not None and assay.features_path.exists():
            feats = pd.read_parquet(assay.features_path)
            feature_ids = feats["feature_id"].astype(str).tolist()
            var = feats.set_index("feature_id", drop=False)
            var.index.name = None
        else:
            feats = None
            var = None

        if assay.kind == "counts":
            long = pd.read_parquet(assay.data_path)
            X, unit_ids, feature_ids = _counts_long_to_wide(long, units.index, feature_ids)
        elif assay.kind == "continuous":
            wide = pd.read_parquet(assay.data_path).set_index("unit_id")
            wide = wide.reindex(units.index)
            feature_ids = feature_ids or list(wide.columns)
            wide = wide[feature_ids]
            X = wide.to_numpy(dtype=np.float32)
        elif assay.kind == "binary":
            wide = pd.read_parquet(assay.data_path).set_index("unit_id")
            wide = wide.reindex(units.index)
            feature_ids = feature_ids or list(wide.columns)
            wide = wide[feature_ids]
            X = wide.to_numpy(dtype=np.float32)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown assay kind {assay.kind}")

        if var is None:
            var = pd.DataFrame({"feature_id": feature_ids}).set_index("feature_id", drop=False)
            var.index.name = None
        else:
            var = var.reindex(feature_ids)

        ad_obj = ad.AnnData(
            X=X,
            obs=units.copy(),
            var=var,
        )
        ad_obj.uns["assay_kind"] = assay.kind

        # feature structure attachments
        if assay.tree_path is not None and assay.tree_path.exists():
            names, C = tree_variance_covariance(assay.tree_path)
            # Align C to feature order
            name_idx = {n: i for i, n in enumerate(names)}
            order = [name_idx[f] for f in feature_ids if f in name_idx]
            if len(order) == len(feature_ids):
                C_aligned = C[np.ix_(order, order)]
            else:
                # fallback: pad/zero - keep but mark
                C_aligned = np.eye(len(feature_ids), dtype=np.float64)
            ad_obj.varm["tree_C"] = C_aligned
            ad_obj.uns["has_tree"] = True
        if assay.graph_edges_path is not None and assay.graph_edges_path.exists():
            edges = pd.read_parquet(assay.graph_edges_path)
            L = graph_laplacian(edges, feature_ids)
            ad_obj.varm["graph_L"] = L
            ad_obj.uns["has_graph"] = True

        modalities[assay.name] = ad_obj

    mud = md.MuData(modalities)
    mud.uns["dataset_root"] = str(manifest.root)
    return mud


def units_frame(mud: md.MuData) -> pd.DataFrame:
    """Return the unit manifest from a loaded MuData (taken from any modality)."""
    if not mud.mod:
        raise ValueError("MuData has no modalities")
    name = next(iter(mud.mod))
    return mud[name].obs.copy()


def assay_matrix(mud: md.MuData, name: str) -> np.ndarray:
    return np.asarray(mud[name].X)
