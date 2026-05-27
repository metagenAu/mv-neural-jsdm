from __future__ import annotations

import torch

from mvnjsdm.models.hierarchy import HierarchyBlock, LevelSpec, empty_hierarchy


def test_empty_hierarchy_zero_offset():
    b = empty_hierarchy(K=4)
    mu, lv = b({"x": torch.zeros(3, dtype=torch.long)})
    assert torch.all(mu == 0.0)
    # logvar is highly negative (negligible variance)
    assert torch.all(lv < -10)


def test_hierarchy_block_pooling():
    levels = [LevelSpec(name="site", n_groups=3, mode="hierarchical_prior")]
    block = HierarchyBlock(levels, K_joint=4)
    ids = torch.tensor([0, 1, 2, 1])
    mu, lv = block({"site": ids})
    assert mu.shape == (4, 4)
    # same id -> same offset
    assert torch.allclose(mu[1], mu[3])


def test_hierarchy_block_off_mode():
    levels = [LevelSpec(name="site", n_groups=3, mode="off")]
    block = HierarchyBlock(levels, K_joint=4)
    mu, lv = block({"site": torch.tensor([0, 1])})
    assert torch.all(mu == 0)
