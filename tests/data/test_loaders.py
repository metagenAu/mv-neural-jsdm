from __future__ import annotations

from pathlib import Path

from mvnjsdm.data.loaders import load_dataset


def test_load_synthetic(synthetic_smoke_dir: Path, synthetic_smoke_truth: dict):
    mud = load_dataset(synthetic_smoke_dir)
    assert set(mud.mod.keys()) == {"assay_a", "assay_b"}
    assert mud["assay_a"].X.shape[0] == synthetic_smoke_truth["n_units"]
    assert mud["assay_a"].X.shape[1] > 0
    assert "tree_C" in mud["assay_a"].varm
