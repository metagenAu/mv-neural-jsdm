"""Write an R-friendly CSV of latents + metadata for SEM / lavaan workflows.

Column renaming convention:

* ``__`` -> ``_`` (R prefers single underscores in column names);
* any leading underscores after rewriting are stripped;
* user-supplied ``rename`` mapping is applied last, overriding the defaults.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _sanitise(name: str) -> str:
    out = name.replace("__", "_")
    while out.startswith("_"):
        out = out[1:]
    return out


def to_sem_csv(
    df_latents: pd.DataFrame,
    path: str | Path,
    *,
    include_columns: list[str] | None = None,
    rename: dict[str, str] | None = None,
) -> None:
    """Write a CSV suited to R's lavaan / piecewiseSEM workflows.

    Parameters
    ----------
    df_latents
        DataFrame from ``export_latents``.
    path
        Destination CSV path.
    include_columns
        Restrict output to ``unit_id`` + these columns. ``None`` keeps all.
    rename
        Explicit name overrides applied after the default ``__`` -> ``_``
        rewrite.
    """
    df = df_latents.copy()
    if include_columns is not None:
        cols = ["unit_id"] + [c for c in include_columns if c in df.columns and c != "unit_id"]
        df = df[cols]
    new_names = {c: _sanitise(c) for c in df.columns}
    if rename:
        for k, v in rename.items():
            if k in df.columns:
                new_names[k] = v
            elif new_names.get(k) is not None:
                # allow keys to refer to either pre- or post-sanitisation names
                for orig, san in new_names.items():
                    if san == k:
                        new_names[orig] = v
    df = df.rename(columns=new_names)
    df.to_csv(Path(path), index=False)
