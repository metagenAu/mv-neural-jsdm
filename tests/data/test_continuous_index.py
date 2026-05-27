from __future__ import annotations

import numpy as np
import pandas as pd

from mvnjsdm.data.continuous_index import ContinuousIndexScaler


def test_continuous_index_scaler():
    df = pd.DataFrame(
        {
            "unit_id": ["a", "b", "c", "d"],
            "cont__doy": [0.0, 10.0, 20.0, 30.0],
            "other": [1, 2, 3, 4],
        }
    )
    s = ContinuousIndexScaler().fit(df)
    out = s.transform(df)
    assert np.isclose(out["cont__doy"].mean(), 0.0, atol=1e-6)
    assert np.isclose(out["cont__doy"].std(), 1.0, atol=0.2)
    assert (out["other"] == df["other"]).all()
