from __future__ import annotations

import numpy as np

from mvnjsdm.data.transforms import clr, library_size, log1p, zscore_masked


def test_clr_zero_mean():
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
    z = clr(x)
    assert np.allclose(z.mean(axis=-1), 0.0, atol=1e-8)


def test_log1p():
    x = np.array([0.0, 1.0, 2.0])
    assert np.allclose(log1p(x), np.log1p(x))


def test_library_size():
    x = np.arange(12).reshape(3, 4)
    assert np.array_equal(library_size(x), x.sum(axis=-1))


def test_zscore_masked():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    m = np.ones_like(x)
    z = zscore_masked(x, m)
    # column means should be roughly 0
    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-6)
