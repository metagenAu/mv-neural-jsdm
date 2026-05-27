from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from mvnjsdm.data.schema import (
    AssayConfig,
    AssayManifest,
    DatasetManifest,
    batch_cols,
    cont_cols,
    env_cols,
    group_cols,
    validate_units_frame,
)


def test_column_helpers():
    df = pd.DataFrame(
        {
            "unit_id": ["a", "b"],
            "group__site": ["x", "y"],
            "env__temp": [1.0, 2.0],
            "cont__doy": [10, 20],
            "batch__run": ["r1", "r2"],
            "other": [0, 0],
        }
    )
    assert group_cols(df) == ["group__site"]
    assert env_cols(df) == ["env__temp"]
    assert cont_cols(df) == ["cont__doy"]
    assert batch_cols(df) == ["batch__run"]


def test_validate_units_frame_dup():
    df = pd.DataFrame({"unit_id": ["a", "a"]})
    with pytest.raises(ValueError):
        validate_units_frame(df)


def test_assay_config_pydantic():
    cfg = AssayConfig(name="a", kind="counts", likelihood="nb")
    assert cfg.size_factor is False
    assert cfg.transform == "identity"


def test_discover_dataset(synthetic_smoke_dir: Path):
    m = DatasetManifest.discover(synthetic_smoke_dir)
    names = sorted(a.name for a in m.assays)
    assert names == ["assay_a", "assay_b"]
    for a in m.assays:
        assert a.kind == "counts"
        assert a.tree_path is not None
