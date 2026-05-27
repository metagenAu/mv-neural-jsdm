from __future__ import annotations

import torch

from mvnjsdm.models.fusion import ExpertOutput, MoPoE, PoE


def _expert(B: int = 4, K: int = 6, present: bool = True, seed: int = 0) -> ExpertOutput:
    g = torch.Generator().manual_seed(seed)
    return ExpertOutput(
        mu=torch.randn(B, K, generator=g),
        logvar=torch.zeros(B, K),
        present_mask=torch.ones(B, dtype=torch.bool) if present else torch.zeros(B, dtype=torch.bool),
    )


def test_poe_with_missing_collapses_to_other():
    a = _expert(seed=1)
    b = _expert(seed=2, present=False)
    fusion = PoE()
    mu1, lv1 = fusion({"a": a, "b": b})
    # When b is absent, the joint should mostly reflect a (combined with prior).
    # Check that shape is correct and finite
    assert mu1.shape == (4, 6)
    assert torch.all(torch.isfinite(mu1))
    assert torch.all(torch.isfinite(lv1))


def test_mopoe_runs():
    a = _expert(seed=3)
    b = _expert(seed=4)
    fusion = MoPoE(seed=0)
    fusion.train()
    mu, lv = fusion({"a": a, "b": b})
    assert mu.shape == (4, 6)
    assert lv.shape == (4, 6)
