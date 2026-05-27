from __future__ import annotations

import numpy as np
import pytest

from mvnjsdm.analysis.interaction_matrix import (
    cross_assay_interaction_matrix,
    latent_interaction_matrix,
    within_assay_interaction_matrix,
)


@pytest.mark.integration
def test_latent_interaction_matrix_shape_and_symmetry(trained_smoke):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]
    M = latent_interaction_matrix(model, dm, split="all")
    K = model.joint_dim
    assert M.shape == (K, K)
    assert np.allclose(M, M.T, atol=1e-5)


@pytest.mark.integration
def test_cross_assay_interaction_matrix_shape(trained_smoke):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]
    df = cross_assay_interaction_matrix(
        model, dm, assay_a="assay_a", assay_b="assay_b", split="all"
    )
    F_a = dm.assay_shapes["assay_a"]
    F_b = dm.assay_shapes["assay_b"]
    assert df.shape == (F_a, F_b)


@pytest.mark.integration
def test_cross_assay_interaction_matrix_linear_fallback(trained_smoke):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]
    df = cross_assay_interaction_matrix(
        model, dm, assay_a="assay_a", assay_b="assay_b", use_jacobian=False
    )
    F_a = dm.assay_shapes["assay_a"]
    F_b = dm.assay_shapes["assay_b"]
    assert df.shape == (F_a, F_b)


@pytest.mark.integration
def test_within_assay_interaction_matrix(trained_smoke):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]
    df = within_assay_interaction_matrix(model, dm, assay="assay_a", split="all")
    F_a = dm.assay_shapes["assay_a"]
    assert df.shape == (F_a, F_a)
    M = df.to_numpy()
    assert np.allclose(M, M.T, atol=1e-5)
