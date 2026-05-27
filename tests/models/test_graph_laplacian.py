"""GraphLaplacian feature prior tests."""

from __future__ import annotations

import numpy as np
import torch

from mvnjsdm.models.feature_priors import GraphLaplacian


def _chain_laplacian(n: int) -> np.ndarray:
    """Combinatorial Laplacian of a 1-D chain of ``n`` nodes."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    D = np.diag(A.sum(axis=1))
    return D - A


def test_chain_laplacian_prefers_smooth_over_rough():
    L = _chain_laplacian(6)
    prior = GraphLaplacian(L, init_alpha=1.0)

    # smooth: a linear ramp
    W_smooth = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]], dtype=torch.float32)
    # rough: high-frequency alternation with the same overall scale
    W_rough = torch.tensor([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]], dtype=torch.float32)

    lp_smooth = prior.log_prob(W_smooth).item()
    lp_rough = prior.log_prob(W_rough).item()
    assert lp_smooth > lp_rough


def test_graph_laplacian_pulls_W_toward_smooth_during_training():
    """Tiny linear model: y = X @ W. Targets are smooth along the chain;
    prior should make learned W smoother than without the prior."""
    torch.manual_seed(0)
    n_features = 6
    n_units = 50
    L = _chain_laplacian(n_features)

    # Random inputs and a noisy smooth target
    X = torch.randn(n_units, 4)
    W_true_smooth = torch.tensor(
        [[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
         [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
         [-0.3, -0.1, 0.1, 0.3, 0.5, 0.7]],
        dtype=torch.float32,
    )
    Y = X @ W_true_smooth + 0.5 * torch.randn(n_units, n_features)

    def _train(use_prior: bool, alpha: float = 5.0) -> torch.Tensor:
        torch.manual_seed(0)
        W = torch.randn(4, n_features, requires_grad=True)
        prior = GraphLaplacian(L, init_alpha=alpha) if use_prior else None
        opt = torch.optim.Adam([W], lr=0.05)
        for _ in range(200):
            opt.zero_grad()
            pred = X @ W
            mse = ((pred - Y) ** 2).mean()
            loss = mse
            if prior is not None:
                # we want to MAXIMISE log_prob, so subtract it
                loss = loss - prior.log_prob(W) / (n_units * n_features)
            loss.backward()
            opt.step()
        return W.detach()

    W_with = _train(use_prior=True)
    W_without = _train(use_prior=False)

    # measure roughness: tr(W L W^T)
    L_t = torch.as_tensor(L, dtype=torch.float32)
    rough_with = (W_with @ L_t * W_with).sum().item()
    rough_without = (W_without @ L_t * W_without).sum().item()
    assert rough_with < rough_without
