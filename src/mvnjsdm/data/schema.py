"""Schemas for the mv-neural-jsdm dataset layout.

Domain-agnostic. Column naming conventions:

    group__<level>   categorical grouping factor (e.g. "group__site")
    env__<name>      environmental covariate (numeric)
    cont__<name>     continuous index (numeric; e.g. time or position)
    batch__<name>    batch/technical covariate (categorical)

The unit manifest (``units.parquet``) carries one row per unit_id with all of
the above prefixed columns.

Per-assay tables follow:
    assay__<name>__counts.parquet   long: unit_id, feature_id, count
    assay__<name>__values.parquet   wide: unit_id + feature cols (continuous)
    assay__<name>__binary.parquet   wide: unit_id + feature cols (0/1)
    assay__<name>__features.parquet feature_id + annotation cols
    assay__<name>__tree.newick      optional Newick tree
    assay__<name>__graph.edges.parquet optional edge list
    assay__<name>__taxonomy.parquet optional taxonomy table
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

AssayKind = Literal["counts", "continuous", "binary"]


class AssayManifest(BaseModel):
    """Describes the files associated with one assay."""

    name: str
    kind: AssayKind
    data_path: Path
    features_path: Path | None = None
    tree_path: Path | None = None
    graph_edges_path: Path | None = None
    taxonomy_path: Path | None = None

    @field_validator("name")
    @classmethod
    def _no_dunder_in_name(cls, v: str) -> str:
        if "__" in v:
            raise ValueError("assay name must not contain '__'")
        return v


class DatasetManifest(BaseModel):
    """Top-level dataset manifest. Resolved paths are relative to ``root``."""

    root: Path
    units_path: Path
    assays: list[AssayManifest] = Field(default_factory=list)

    @classmethod
    def discover(cls, root: str | Path) -> DatasetManifest:
        """Discover a dataset on disk by following the prefix conventions."""
        root = Path(root)
        units_path = root / "units.parquet"
        if not units_path.exists():
            raise FileNotFoundError(f"units.parquet missing under {root}")

        assays: list[AssayManifest] = []
        for path in sorted(root.iterdir()):
            stem = path.name
            if not stem.startswith("assay__"):
                continue
            parts = stem.split("__")
            # assay__<name>__<role>.<ext>
            if len(parts) < 3:
                continue
            name = parts[1]
            role = "__".join(parts[2:])
            # only register data-bearing tables here
            data_kind: AssayKind | None = None
            if role.endswith("counts.parquet"):
                data_kind = "counts"
            elif role.endswith("values.parquet"):
                data_kind = "continuous"
            elif role.endswith("binary.parquet"):
                data_kind = "binary"
            else:
                continue
            features_path = root / f"assay__{name}__features.parquet"
            tree_path = root / f"assay__{name}__tree.newick"
            graph_edges_path = root / f"assay__{name}__graph.edges.parquet"
            taxonomy_path = root / f"assay__{name}__taxonomy.parquet"
            assays.append(
                AssayManifest(
                    name=name,
                    kind=data_kind,
                    data_path=path,
                    features_path=features_path if features_path.exists() else None,
                    tree_path=tree_path if tree_path.exists() else None,
                    graph_edges_path=graph_edges_path if graph_edges_path.exists() else None,
                    taxonomy_path=taxonomy_path if taxonomy_path.exists() else None,
                )
            )

        return cls(root=root, units_path=units_path, assays=assays)


# ----- column helpers -------------------------------------------------------

GROUP_PREFIX = "group__"
ENV_PREFIX = "env__"
CONT_PREFIX = "cont__"
BATCH_PREFIX = "batch__"


def group_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(GROUP_PREFIX)]


def env_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(ENV_PREFIX)]


def cont_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(CONT_PREFIX)]


def batch_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(BATCH_PREFIX)]


def validate_units_frame(df: pd.DataFrame) -> None:
    if "unit_id" not in df.columns:
        raise ValueError("units frame missing 'unit_id'")
    if df["unit_id"].duplicated().any():
        raise ValueError("duplicate unit_id in units frame")


# ----- assay/likelihood config (pydantic) -----------------------------------


class AssayConfig(BaseModel):
    """Per-assay runtime configuration."""

    name: str
    kind: AssayKind
    likelihood: Literal["nb", "zinb", "gaussian_masked", "bernoulli", "poisson"]
    encoder_config: dict = Field(default_factory=dict)
    decoder_config: dict = Field(default_factory=dict)
    feature_structure_config: dict | None = None
    size_factor: bool = False
    transform: Literal["clr", "log1p", "identity", "zscore_masked"] = "identity"
