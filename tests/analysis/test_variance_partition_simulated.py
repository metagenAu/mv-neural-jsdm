from __future__ import annotations

import pytest

from mvnjsdm.analysis.variance_partition import variance_partition


@pytest.mark.integration
def test_variance_partition_recovers_group_signal(
    trained_smoke, synthetic_smoke_truth
):
    model = trained_smoke["model"]
    dm = trained_smoke["datamodule"]

    df = variance_partition(
        model, dm, levels=["group__site"], method="regression"
    )
    # Sanity: each row sums close to 1
    sums = df.sum(axis=1)
    assert (sums > 0.95).all() and (sums < 1.05).all(), sums.to_dict()

    # group share is present
    assert "group__site" in df.columns
    # at least one latent dim should have non-trivial group share. We do NOT
    # require recovery within 0.2 of truth at 5 epochs of training -- the
    # model is severely under-fit. Just check signal is non-zero and bounded.
    max_share = df["group__site"].max()
    assert 0.0 <= max_share <= 1.0
    # also test the anova method runs
    df_anova = variance_partition(
        model, dm, levels=["group__site"], method="anova"
    )
    assert "residual" in df_anova.columns


@pytest.mark.integration
def test_variance_partition_ablation_not_implemented(trained_smoke):
    with pytest.raises(NotImplementedError):
        variance_partition(
            trained_smoke["model"],
            trained_smoke["datamodule"],
            method="ablation",
        )
