"""Tests for cross-assay coupling analyses."""

from __future__ import annotations

import pytest

from mvnjsdm.analysis.coupling import (
    contemporaneous_coupling,
    lagged_coupling,
    latent_lagged_covariance,
)


@pytest.mark.integration
def test_contemporaneous_coupling_shape_and_finite(trained_smoke):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]
    df = contemporaneous_coupling(
        model, dm, assay_a="assay_a", assay_b="assay_b", n_samples=4
    )
    F_a = dm.assay_shapes["assay_a"]
    F_b = dm.assay_shapes["assay_b"]
    assert df.shape == (F_a, F_b)
    assert df.notna().all().all()
    # Pearson is in [-1, 1].
    assert df.to_numpy().min() >= -1.0 - 1e-6
    assert df.to_numpy().max() <= 1.0 + 1e-6


@pytest.mark.integration
def test_lagged_coupling_errors_without_gp(trained_smoke):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]
    with pytest.raises(RuntimeError, match="GP"):
        lagged_coupling(
            model, dm, assay_a="assay_a", assay_b="assay_b",
            index_name="cont__day_of_year", lag=1.0,
        )
    with pytest.raises(RuntimeError, match="GP"):
        latent_lagged_covariance(model, dm, index_name="cont__day_of_year", lag=1.0)


@pytest.mark.integration
def test_lagged_coupling_zero_lag_matches_contemporaneous_shape(trained_smoke):
    """At lag=0 the function falls back to contemporaneous, returning the same shape.
    Works even without GP because the zero-lag branch short-circuits."""
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]
    df = lagged_coupling(
        model, dm, assay_a="assay_a", assay_b="assay_b",
        index_name="cont__day_of_year", lag=0.0, n_samples=4,
    )
    F_a = dm.assay_shapes["assay_a"]
    F_b = dm.assay_shapes["assay_b"]
    assert df.shape == (F_a, F_b)
