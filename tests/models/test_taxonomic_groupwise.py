"""TaxonomicGroupwise feature prior tests."""

from __future__ import annotations

import torch

from mvnjsdm.models.feature_priors import TaxonomicGroupwise


def test_groupwise_prefers_concentrated_over_spread():
    """Two groups of three features. A W that puts mass in exactly one group
    has fewer non-zero group norms than a W that spreads mass across both."""
    group_assignment = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    prior = TaxonomicGroupwise(group_assignment, init_alpha=1.0)

    # concentrated: all weight in group 0
    W_conc = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    # spread: same total l2 norm but split across both groups
    W_spread = torch.tensor([[0.707, 0.707, 0.0, 0.707, 0.707, 0.0]], dtype=torch.float32)

    lp_conc = prior.log_prob(W_conc).item()
    lp_spread = prior.log_prob(W_spread).item()
    # higher log_prob (less negative) is better; concentrated should win.
    assert lp_conc > lp_spread


def test_groupwise_zero_on_zero_input():
    group_assignment = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    prior = TaxonomicGroupwise(group_assignment, init_alpha=2.0)
    W = torch.zeros(3, 4)
    lp = prior.log_prob(W).item()
    # log_prob is -alpha * sum group norms; with zero W the group norms are
    # approximately zero (modulo clamp epsilon), so log_prob should be near 0.
    assert abs(lp) < 1e-4
