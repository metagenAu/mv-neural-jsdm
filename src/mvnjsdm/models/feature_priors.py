"""Feature-structure priors over decoder loading matrices W in R^{K x F}.

Convention: ``W[k, :]`` is the "trait vector" for latent dim k across features.

Pagel's lambda (row-wise):
    W[k, :] ~ N(0, sigma^2 (lambda * C + (1 - lambda) * I))
    rows independent across k.

Brownian: lambda = 1.

Both implement closed-form log density per row, summed over rows.
"""

from __future__ import annotations

import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeatureStructurePrior(nn.Module, abc.ABC):
    @abc.abstractmethod
    def log_prob(self, W: Tensor) -> Tensor:
        """Scalar log-probability of W under this prior."""

    def applies_to(self, assay_name: str) -> bool:  # noqa: D401
        return True


class NoFeaturePrior(FeatureStructurePrior):
    def log_prob(self, W: Tensor) -> Tensor:
        return torch.zeros((), device=W.device, dtype=W.dtype)


class _PhyloBase(FeatureStructurePrior):
    """Holds the tree covariance C and shared MVN log-prob machinery."""

    def __init__(self, C: np.ndarray, learn_sigma: bool = True) -> None:
        super().__init__()
        C_t = torch.as_tensor(C, dtype=torch.float32)
        F_ = C_t.shape[0]
        # add tiny jitter
        self.register_buffer("C", C_t + 1e-6 * torch.eye(F_))
        self.register_buffer("I", torch.eye(F_))
        self.F = F_
        # learnable log_sigma^2 (positive via softplus)
        if learn_sigma:
            self.raw_sigma2 = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("raw_sigma2", torch.zeros(()))

    def sigma2(self) -> Tensor:
        return F.softplus(self.raw_sigma2) + 1e-6

    def _row_mvn_logprob(self, W: Tensor, Sigma: Tensor) -> Tensor:
        # W: [K, F], Sigma: [F, F]
        # Use Cholesky for stability.
        L = torch.linalg.cholesky(Sigma)
        # logdet
        logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
        # solve L y = W^T (column-major)
        # For each row k: x^T Sigma^{-1} x = ||L^{-1} x||^2
        # Stack rows: solve_triangular(L, W^T) gives [F, K]; squared norm column-wise.
        y = torch.linalg.solve_triangular(L, W.transpose(0, 1), upper=False)
        quad = (y ** 2).sum(dim=0)  # [K]
        K = W.shape[0]
        F_ = W.shape[1]
        const = -0.5 * F_ * np.log(2.0 * np.pi)
        lp = K * (const - 0.5 * logdet) - 0.5 * quad.sum()
        return lp


class PhylogeneticBrownian(_PhyloBase):
    """W[k,:] ~ N(0, sigma^2 * C)."""

    def log_prob(self, W: Tensor) -> Tensor:
        Sigma = self.sigma2() * self.C
        return self._row_mvn_logprob(W, Sigma)


class PhylogeneticPagel(_PhyloBase):
    """W[k,:] ~ N(0, sigma^2 * (lambda C + (1-lambda) I))."""

    def __init__(self, C: np.ndarray, learn_sigma: bool = True, init_lambda: float = 0.5) -> None:
        super().__init__(C, learn_sigma=learn_sigma)
        # raw_lambda -> sigmoid -> [0, 1]
        init_raw = float(np.log(init_lambda / (1 - init_lambda + 1e-9) + 1e-9))
        self.raw_lambda = nn.Parameter(torch.tensor(init_raw))

    def lambda_value(self) -> Tensor:
        return torch.sigmoid(self.raw_lambda)

    def log_prob(self, W: Tensor) -> Tensor:
        lam = self.lambda_value()
        Sigma = self.sigma2() * (lam * self.C + (1.0 - lam) * self.I)
        return self._row_mvn_logprob(W, Sigma)


class GraphLaplacian(FeatureStructurePrior):
    """Tikhonov-style regulariser using a feature graph Laplacian L.

    Defines an improper Gaussian-like prior on each row of W with precision
    proportional to L (plus an optional ridge):

        log_prob(W) = sum_k -alpha * W[k,:] @ L @ W[k,:]

    The normalising constant is dropped because L is typically rank-deficient
    (connected graphs have a zero eigenvalue from the constant vector). This
    means the result is a regulariser, not a proper density.

    ``alpha`` is a learnable positive scalar (softplus on an unconstrained
    parameter).
    """

    def __init__(self, L: Tensor | np.ndarray, init_alpha: float = 1.0) -> None:
        super().__init__()
        L_t = torch.as_tensor(L, dtype=torch.float32)
        if L_t.ndim != 2 or L_t.shape[0] != L_t.shape[1]:
            raise ValueError("L must be a square 2-D matrix")
        self.register_buffer("L", L_t)
        # softplus^{-1}(init_alpha): solve softplus(x) = init_alpha
        # x = log(exp(init_alpha) - 1)
        init_raw = float(np.log(np.expm1(max(init_alpha, 1e-4))))
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    def alpha(self) -> Tensor:
        return F.softplus(self.raw_alpha) + 1e-6

    def log_prob(self, W: Tensor) -> Tensor:
        # W: [K, F]; quadratic form sum_k W[k,:] L W[k,:]^T
        L = self.L
        # Wt @ L: [K, F]; element-wise W * (W @ L): sum_k row-wise quadratic form
        WL = W @ L  # [K, F]
        quad = (W * WL).sum()
        return -self.alpha() * quad

    def applies_to(self, assay_name: str) -> bool:  # noqa: D401
        return True


class TaxonomicGroupwise(FeatureStructurePrior):
    """Group-sparse prior on loading rows.

    Given ``group_assignment[f] = g`` mapping each feature to a group id, the
    log-prob is the negative group-l2 sum:

        log_prob(W) = -alpha * sum_k sum_g ||W[k, group==g]||_2

    This encourages each latent dim to use a few groups rather than spreading
    weight uniformly across groups (an l_{2,1}-type sparsity).
    """

    def __init__(
        self,
        group_assignment: Tensor | np.ndarray,
        init_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        ga = torch.as_tensor(group_assignment, dtype=torch.long)
        if ga.ndim != 1:
            raise ValueError("group_assignment must be 1-D")
        self.register_buffer("group_assignment", ga)
        n_groups = int(ga.max().item()) + 1 if ga.numel() > 0 else 0
        # Pre-compute a [G, F] indicator for vectorised group sums.
        F_ = ga.shape[0]
        ind = torch.zeros(n_groups, F_, dtype=torch.float32)
        if F_ > 0:
            ind[ga, torch.arange(F_)] = 1.0
        self.register_buffer("group_indicator", ind)
        init_raw = float(np.log(np.expm1(max(init_alpha, 1e-4))))
        self.raw_alpha = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))

    def alpha(self) -> Tensor:
        return F.softplus(self.raw_alpha) + 1e-6

    def log_prob(self, W: Tensor) -> Tensor:
        # W: [K, F]; per (k, group) compute ||W[k, group]||_2 then sum.
        W2 = W ** 2  # [K, F]
        # sum within each group: [K, G] = W2 @ group_indicator^T
        sums = W2 @ self.group_indicator.t()  # [K, G]
        norms = torch.sqrt(sums.clamp_min(1e-12))
        return -self.alpha() * norms.sum()

    def applies_to(self, assay_name: str) -> bool:  # noqa: D401
        return True
