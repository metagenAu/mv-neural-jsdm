"""Join decoder loadings against external feature annotations.

Returns a long-form table for plotting / filtering: one row per
(latent_dim, feature_id) pair with the loading magnitude and any annotation
columns attached.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annotate_loadings(
    loadings,
    feature_metadata: pd.DataFrame,
    *,
    on: str = "feature_id",
) -> pd.DataFrame:
    """Long-form join of decoder loadings with feature metadata.

    Parameters
    ----------
    loadings
        Either a numpy array of shape ``[K, F]`` -- in which case the
        feature_metadata's ``on`` column is used to assign feature ids in
        column order -- or a DataFrame indexed by feature_id with one column
        per latent dim.
    feature_metadata
        DataFrame containing an ``on`` column plus arbitrary annotation
        columns to attach.
    on
        Name of the feature id column.

    Returns
    -------
    DataFrame with columns ``feature_id``, ``latent_dim`` (int), ``loading``,
    plus any annotation columns from ``feature_metadata``.
    """
    if isinstance(loadings, np.ndarray):
        if on not in feature_metadata.columns:
            raise ValueError(f"feature_metadata missing column '{on}'")
        fids = feature_metadata[on].astype(str).tolist()
        K, F = loadings.shape
        if F != len(fids):
            raise ValueError(
                f"loadings has {F} features but feature_metadata has {len(fids)}"
            )
        df_load = pd.DataFrame(
            loadings.T, index=fids, columns=[f"latent_{k}" for k in range(K)]
        )
        df_load.index.name = on
    elif isinstance(loadings, pd.DataFrame):
        df_load = loadings.copy()
        if df_load.index.name is None:
            df_load.index.name = on
    else:
        raise TypeError(f"unsupported loadings type: {type(loadings)}")

    long = df_load.reset_index().melt(
        id_vars=[df_load.index.name],
        var_name="latent_dim",
        value_name="loading",
    )
    long["latent_dim"] = (
        long["latent_dim"].astype(str).str.replace("latent_", "", regex=False).astype(int)
    )

    merged = long.merge(feature_metadata, on=on, how="left")
    return merged
