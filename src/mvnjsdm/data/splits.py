"""Train/val/test splits by group level."""

from __future__ import annotations

import numpy as np
import pandas as pd


def by_group_split(
    units: pd.DataFrame,
    group_col: str,
    val_frac: float = 0.15,
    test_frac: float = 0.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Return index arrays (positions, not unit ids) for train/val/test.

    Splits whole groups (i.e. all units of a given group end up in one split).
    If ``group_col`` is missing, falls back to a row-wise random split.
    """
    rng = np.random.default_rng(seed)
    if group_col not in units.columns:
        idx = np.arange(len(units))
        rng.shuffle(idx)
        n_test = int(round(test_frac * len(idx)))
        n_val = int(round(val_frac * len(idx)))
        test = idx[:n_test]
        val = idx[n_test : n_test + n_val]
        train = idx[n_test + n_val :]
        return {"train": train, "val": val, "test": test}

    groups = units[group_col].astype(str).to_numpy()
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    n_test = int(round(test_frac * len(uniq)))
    n_val = max(1, int(round(val_frac * len(uniq))))
    test_g = set(uniq[:n_test].tolist())
    val_g = set(uniq[n_test : n_test + n_val].tolist())

    train, val, test = [], [], []
    for i, g in enumerate(groups):
        if g in test_g:
            test.append(i)
        elif g in val_g:
            val.append(i)
        else:
            train.append(i)
    return {
        "train": np.array(train, dtype=np.int64),
        "val": np.array(val, dtype=np.int64),
        "test": np.array(test, dtype=np.int64),
    }
