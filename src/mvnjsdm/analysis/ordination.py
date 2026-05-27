"""Lightweight ordination helpers for exported latents.

Currently provides PCA via numpy SVD; UMAP / t-SNE are left for users to apply
externally on the export (``umap-learn`` is intentionally not a hard dep).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def latent_pca(
    df_latents: pd.DataFrame,
    n_components: int = 2,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Add ``PC1..PCn`` columns to ``df_latents`` via SVD-based PCA.

    Parameters
    ----------
    df_latents
        DataFrame as returned by ``export_latents``.
    n_components
        Number of principal components to compute. Clamped to
        ``min(n_components, n_features, n_units)``.
    columns
        Which columns to PCA. Defaults to all ``latent_*`` columns.

    Returns
    -------
    A copy of ``df_latents`` with added ``PC1..PCn`` numeric columns.
    """
    df = df_latents.copy()
    if columns is None:
        columns = [c for c in df.columns if c.startswith("latent_")]
    if not columns:
        raise ValueError("no latent columns found")

    X = df[columns].to_numpy(dtype=np.float64)
    X = X - X.mean(0, keepdims=True)
    n_components = min(n_components, X.shape[1], X.shape[0])
    # SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    PC = U[:, :n_components] * S[:n_components]
    for i in range(n_components):
        df[f"PC{i + 1}"] = PC[:, i]
    return df


# TODO: UMAP / t-SNE wrappers. Users can run umap-learn directly on the
# exported latent columns; we intentionally don't take a hard umap dep.
