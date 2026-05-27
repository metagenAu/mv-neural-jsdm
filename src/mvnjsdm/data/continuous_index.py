"""Helpers for ``cont__<name>`` continuous index columns.

These columns are centred/scaled per-key so that downstream priors (GP, AR)
operate on a common scale. The fit step records mean/std; the transform step
applies them.
"""

from __future__ import annotations

import pandas as pd

from .schema import cont_cols


class ContinuousIndexScaler:
    def __init__(self) -> None:
        self.stats: dict[str, tuple[float, float]] = {}

    def fit(self, df: pd.DataFrame) -> ContinuousIndexScaler:
        for c in cont_cols(df):
            vals = df[c].astype(float).to_numpy()
            mu = float(vals.mean())
            sigma = float(vals.std())
            if sigma < 1e-9:
                sigma = 1.0
            self.stats[c] = (mu, sigma)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, (mu, sigma) in self.stats.items():
            if c in out.columns:
                out[c] = (out[c].astype(float) - mu) / sigma
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
