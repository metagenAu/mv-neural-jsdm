"""Per-assay likelihoods. Smoke path: NB only."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LikelihoodParams:
    """Container for decoder outputs. Only ``mu`` is required.

    For NB: ``mu`` is the rate, ``theta`` the dispersion (inverse).
    For ZINB: also ``gate``.
    For Gaussian: ``log_sigma``.
    """

    mu: Tensor
    theta: Tensor | None = None
    log_sigma: Tensor | None = None
    gate: Tensor | None = None


class Likelihood(abc.ABC):
    @abc.abstractmethod
    def log_prob(self, x: Tensor, params: LikelihoodParams, mask: Tensor | None = None) -> Tensor:
        """Returns per-sample log-prob (sum over features), shape [B]."""


def _nb_log_prob(x: Tensor, mu: Tensor, theta: Tensor) -> Tensor:
    """Element-wise NB log-prob using (mu, theta) parameterisation.

    p(x | mu, theta) = Gamma(x+theta)/(Gamma(theta) x!) * (theta/(theta+mu))^theta * (mu/(theta+mu))^x

    Returns tensor of same shape as x.
    """
    eps = 1e-8
    theta = theta.clamp_min(eps)
    mu = mu.clamp_min(eps)
    log_theta_mu = torch.log(theta + mu)
    return (
        torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - log_theta_mu)
        + x * (torch.log(mu) - log_theta_mu)
    )


class NBLikelihood(Likelihood):
    def log_prob(self, x: Tensor, params: LikelihoodParams, mask: Tensor | None = None) -> Tensor:
        assert params.theta is not None, "NB requires theta"
        lp = _nb_log_prob(x, params.mu, params.theta)
        if mask is not None:
            lp = lp * mask
        return lp.sum(dim=-1)


class ZINBLikelihood(Likelihood):  # stub
    def log_prob(self, x, params, mask=None):  # pragma: no cover - stub
        raise NotImplementedError("ZINBLikelihood is a stub")


class GaussianMaskedLikelihood(Likelihood):  # stub
    def log_prob(self, x, params, mask=None):  # pragma: no cover - stub
        raise NotImplementedError("GaussianMaskedLikelihood is a stub")


class BernoulliLikelihood(Likelihood):  # stub
    def log_prob(self, x, params, mask=None):  # pragma: no cover - stub
        raise NotImplementedError("BernoulliLikelihood is a stub")


class PoissonLikelihood(Likelihood):  # stub
    def log_prob(self, x, params, mask=None):  # pragma: no cover - stub
        raise NotImplementedError("PoissonLikelihood is a stub")


LIKELIHOODS = {
    "nb": NBLikelihood,
    "zinb": ZINBLikelihood,
    "gaussian_masked": GaussianMaskedLikelihood,
    "bernoulli": BernoulliLikelihood,
    "poisson": PoissonLikelihood,
}
