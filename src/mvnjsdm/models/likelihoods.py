"""Per-assay likelihoods. Smoke path: NB only."""

from __future__ import annotations

import abc
import math
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


class ZINBLikelihood(Likelihood):
    """Zero-inflated negative binomial.

    Mixture of a point mass at zero (probability ``gate``) and a NB(mu, theta)
    component. The per-element log-prob is:

        log p(x=0) = log(gate + (1 - gate) * NB(0 | mu, theta))
        log p(x>0) = log(1 - gate) + log NB(x | mu, theta)

    Implemented numerically with :func:`torch.logaddexp` to avoid loss of
    precision when ``gate`` is small or NB(0) is small. ``params.gate`` is
    expected to be a probability in ``[0, 1]`` (the decoder applies the
    sigmoid).
    """

    def log_prob(self, x: Tensor, params: LikelihoodParams, mask: Tensor | None = None) -> Tensor:
        assert params.theta is not None, "ZINB requires theta"
        assert params.gate is not None, "ZINB requires gate"
        eps = 1e-8
        gate = params.gate.clamp(min=eps, max=1.0 - eps)
        log_gate = torch.log(gate)
        log_1m_gate = torch.log(1.0 - gate)
        # NB log-prob for x=0 vs x>0 computed once per cell
        log_nb_x = _nb_log_prob(x, params.mu, params.theta)
        # zero-cell: logaddexp(log_gate, log_1m_gate + log_nb_0)
        zeros = torch.zeros_like(x)
        log_nb_0 = _nb_log_prob(zeros, params.mu, params.theta)
        log_p_zero = torch.logaddexp(log_gate, log_1m_gate + log_nb_0)
        log_p_pos = log_1m_gate + log_nb_x
        is_zero = (x < 0.5).to(log_p_zero.dtype)
        lp = is_zero * log_p_zero + (1.0 - is_zero) * log_p_pos
        if mask is not None:
            lp = lp * mask
        return lp.sum(dim=-1)


class GaussianMaskedLikelihood(Likelihood):
    """N(mu, sigma^2) with sigma = exp(log_sigma) per feature; masked features
    contribute zero log-prob."""

    def log_prob(self, x: Tensor, params: LikelihoodParams, mask: Tensor | None = None) -> Tensor:
        assert params.log_sigma is not None, "Gaussian likelihood requires log_sigma"
        log_sigma = params.log_sigma.clamp(min=-7.0, max=7.0)
        sigma = torch.exp(log_sigma).clamp_min(1e-4)
        lp = (
            -0.5 * ((x - params.mu) / sigma) ** 2
            - log_sigma
            - 0.5 * math.log(2.0 * math.pi)
        )
        if mask is not None:
            lp = lp * mask
        return lp.sum(dim=-1)


class BernoulliLikelihood(Likelihood):
    """Bernoulli with params.mu interpreted as logits; masked entries dropped."""

    def log_prob(self, x: Tensor, params: LikelihoodParams, mask: Tensor | None = None) -> Tensor:
        logits = params.mu
        lp = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")
        if mask is not None:
            lp = lp * mask
        return lp.sum(dim=-1)


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
