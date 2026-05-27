"""Posterior-decomposition variance partitioning.

For each latent dimension ``z_k``, fit a simple linear model expressing
``z_k`` as a function of one or more grouping levels (group__*) and continuous
environmental covariates (env__*), and decompose the total variance into
contributions from each factor + a "residual" share.

Three methods are supported:

* ``regression`` (default): per-dim linear model with one-hot encoded groups
  and standardised env covariates. Variance share for a factor is computed via
  Type-I (sequential) sums of squares in the order the factors are passed in,
  divided by the total SS of ``z_k``. This is the post-hoc decomposition you
  want when factors are correlated; documented as sequential.

* ``anova``: a one-way ANOVA share per categorical level (additive across
  levels). Warned in the docstring -- shares are not orthogonal when groups
  overlap.

* ``ablation``: explicit retraining is required; raises
  ``NotImplementedError`` -- callers should run ``cli.train`` twice with the
  factor toggled and diff the latent variances.

Returns a DataFrame indexed by latent dim (string label) with one column per
requested factor and a final ``residual`` column summing to ~1 per row.

This module does *not* depend on statsmodels or scikit-learn.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .latent_export import export_latents


def _one_hot(series: pd.Series) -> np.ndarray:
    cats = sorted(series.astype(str).unique().tolist())
    if len(cats) <= 1:
        return np.zeros((len(series), 0), dtype=np.float64)
    # drop first category to avoid collinearity with the intercept (which is
    # added as a separate column by the caller)
    mat = np.zeros((len(series), len(cats) - 1), dtype=np.float64)
    s = series.astype(str).to_numpy()
    for j, c in enumerate(cats[1:]):
        mat[:, j] = (s == c).astype(np.float64)
    return mat


def _sequential_ss(
    y: np.ndarray, factor_blocks: list[tuple[str, np.ndarray]]
) -> dict[str, float]:
    """Compute sequential (Type-I) sums of squares for each factor block.

    Each block is added on top of the previous; the marginal increase in
    explained SS attributed to the block is returned. Returns dict with one
    key per factor name plus 'residual'. Values sum to total SS of y.
    """
    n = len(y)
    y_mean = float(np.mean(y))
    ss_total = float(np.sum((y - y_mean) ** 2))

    # Always start with an intercept column
    X = np.ones((n, 1), dtype=np.float64)
    ss_explained_prev = 0.0
    shares: dict[str, float] = {}
    for name, block in factor_blocks:
        if block.shape[1] == 0:
            shares[name] = 0.0
            continue
        X_new = np.concatenate([X, block.astype(np.float64)], axis=1)
        beta, *_ = np.linalg.lstsq(X_new, y, rcond=None)
        y_hat = X_new @ beta
        ss_explained = float(np.sum((y_hat - y_mean) ** 2))
        shares[name] = max(0.0, ss_explained - ss_explained_prev)
        ss_explained_prev = ss_explained
        X = X_new
    shares["residual"] = max(0.0, ss_total - ss_explained_prev)
    # normalise to fractions of total
    if ss_total <= 1e-12:
        return {k: 0.0 for k in shares}
    return {k: v / ss_total for k, v in shares.items()}


def variance_partition(
    model: Any,
    datamodule: Any,
    *,
    levels: list[str] | None = None,
    env_covariates: list[str] | None = None,
    method: Literal["anova", "regression", "ablation"] = "regression",
    split: Literal["train", "val", "all"] = "all",
) -> pd.DataFrame:
    """Decompose posterior-mean latent variance across factors.

    Parameters
    ----------
    model, datamodule
        Trained model + setup data module.
    levels
        Group columns to attribute (e.g. ``["group__site"]``). If ``None``,
        all ``group__*`` columns present in ``datamodule.units`` are used.
    env_covariates
        Continuous env columns (e.g. ``["env__temp"]``). If ``None``, all
        ``env__*`` columns are used.
    method
        One of ``regression`` (sequential SS), ``anova`` (per-level one-way),
        or ``ablation`` (raises NotImplementedError).

    Returns
    -------
    DataFrame indexed by latent dim name with columns for each factor (in the
    order supplied) plus ``residual``. Values are fractions of the per-dim
    variance and sum to ~1 per row (small drift possible from numerical
    issues).
    """
    if method == "ablation":
        raise NotImplementedError(
            "ablation requires retraining; use cli.train with hierarchy=off "
            "or env=none and compare"
        )

    df = export_latents(model, datamodule, split=split, n_samples=1)
    units = df.copy()

    if levels is None:
        levels = [c for c in units.columns if c.startswith("group__")]
    if env_covariates is None:
        env_covariates = [c for c in units.columns if c.startswith("env__")]

    latent_cols = [c for c in units.columns if c.startswith("latent_")]
    if not latent_cols:
        raise ValueError("no latent_* columns in export; train the model first")

    # Build factor blocks (in order)
    factor_blocks: list[tuple[str, np.ndarray]] = []
    for lev in levels:
        if lev not in units.columns:
            factor_blocks.append((lev, np.zeros((len(units), 0))))
            continue
        factor_blocks.append((lev, _one_hot(units[lev])))
    if env_covariates:
        env_mat_cols = [c for c in env_covariates if c in units.columns]
        if env_mat_cols:
            E = units[env_mat_cols].to_numpy(dtype=np.float64)
            # standardise each column (helps numerical conditioning; does not
            # affect SS ratios)
            mu = E.mean(0, keepdims=True)
            sd = E.std(0, keepdims=True)
            sd[sd < 1e-8] = 1.0
            E = (E - mu) / sd
            factor_blocks.append(("env", E))

    out_rows: list[dict[str, Any]] = []
    for lat in latent_cols:
        y = units[lat].to_numpy(dtype=np.float64)
        if method == "regression":
            shares = _sequential_ss(y, factor_blocks)
        elif method == "anova":
            y_mean = float(np.mean(y))
            ss_total = float(np.sum((y - y_mean) ** 2))
            shares = {}
            ss_used = 0.0
            for name, block in factor_blocks:
                if block.shape[1] == 0 or ss_total <= 1e-12:
                    shares[name] = 0.0
                    continue
                X = np.concatenate([np.ones((len(y), 1)), block], axis=1)
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                y_hat = X @ beta
                ss_e = float(np.sum((y_hat - y_mean) ** 2))
                share = max(0.0, ss_e) / ss_total
                shares[name] = share
                ss_used += share
            shares["residual"] = max(0.0, 1.0 - ss_used)
        else:  # pragma: no cover
            raise ValueError(f"unknown method: {method}")
        row = {"latent": lat, **shares}
        out_rows.append(row)

    df_out = pd.DataFrame(out_rows).set_index("latent")
    return df_out
