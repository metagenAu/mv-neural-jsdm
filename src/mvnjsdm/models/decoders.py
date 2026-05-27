"""Decoders. Smoke path: NBDecoder with optional feature-structure prior on W."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .feature_priors import FeatureStructurePrior, NoFeaturePrior
from .likelihoods import LikelihoodParams


class Decoder(nn.Module, abc.ABC):
    feature_prior: FeatureStructurePrior

    @abc.abstractmethod
    def forward(
        self,
        z: Tensor,
        batch_cov: Tensor | None = None,
        size_factor: Tensor | None = None,
    ) -> LikelihoodParams: ...

    def feature_prior_loss(self) -> Tensor:
        return -self.feature_prior.log_prob(self.loading_matrix())

    @abc.abstractmethod
    def loading_matrix(self) -> Tensor:
        """Return the W matrix of shape [K, F] the feature prior acts on."""


class NBDecoder(Decoder):
    """Linear (per-latent) loadings + softplus dispersion.

    Forward path:
        rate = exp(z @ W + b) * size_factor       (shape [B, F])
        theta = softplus(theta_raw) per feature

    The loading matrix ``W`` (shape [K_in, F]) is exposed for the feature
    prior.
    """

    def __init__(
        self,
        in_dim: int,
        n_features: int,
        feature_prior: FeatureStructurePrior | None = None,
        use_size_factor: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_features = n_features
        # small init so initial rate * sf doesn't overflow
        self.W = nn.Parameter(torch.randn(in_dim, n_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_features))
        self.theta_raw = nn.Parameter(torch.zeros(n_features))
        self.feature_prior = feature_prior or NoFeaturePrior()
        self.use_size_factor = use_size_factor

    def loading_matrix(self) -> Tensor:
        return self.W

    def forward(
        self,
        z: Tensor,
        batch_cov: Tensor | None = None,
        size_factor: Tensor | None = None,
    ) -> LikelihoodParams:
        log_rate = z @ self.W + self.b
        log_rate = log_rate.clamp(min=-10.0, max=8.0)
        if self.use_size_factor and size_factor is not None:
            log_rate = log_rate + torch.log(size_factor.clamp_min(1.0)).unsqueeze(-1)
        rate = torch.exp(log_rate.clamp(max=15.0))
        theta = F.softplus(self.theta_raw) + 1e-4
        # broadcast theta to batch
        theta_b = theta.expand_as(rate)
        return LikelihoodParams(mu=rate, theta=theta_b)


class ZINBDecoder(Decoder):
    """Zero-inflated NB decoder.

    Same structure as :class:`NBDecoder`, plus a per-feature ``gate_logits``
    head producing a sigmoid-bounded dropout probability ``gate`` per cell.
    The dropout head is shared across units (a per-feature bias), matching
    the most common ZINB-VAE configuration without unit-specific inputs.
    """

    def __init__(
        self,
        in_dim: int,
        n_features: int,
        feature_prior: FeatureStructurePrior | None = None,
        use_size_factor: bool = True,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_features = n_features
        self.W = nn.Parameter(torch.randn(in_dim, n_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_features))
        self.theta_raw = nn.Parameter(torch.zeros(n_features))
        # per-feature dropout-logit; sigmoid -> gate prob
        self.gate_logits = nn.Parameter(torch.full((n_features,), -2.0))
        self.feature_prior = feature_prior or NoFeaturePrior()
        self.use_size_factor = use_size_factor

    def loading_matrix(self) -> Tensor:
        return self.W

    def forward(
        self,
        z: Tensor,
        batch_cov: Tensor | None = None,
        size_factor: Tensor | None = None,
    ) -> LikelihoodParams:
        log_rate = z @ self.W + self.b
        log_rate = log_rate.clamp(min=-10.0, max=8.0)
        if self.use_size_factor and size_factor is not None:
            log_rate = log_rate + torch.log(size_factor.clamp_min(1.0)).unsqueeze(-1)
        rate = torch.exp(log_rate.clamp(max=15.0))
        theta = F.softplus(self.theta_raw) + 1e-4
        theta_b = theta.expand_as(rate)
        gate = torch.sigmoid(self.gate_logits).expand_as(rate)
        return LikelihoodParams(mu=rate, theta=theta_b, gate=gate)


class GaussianDecoder(Decoder):
    """Linear loadings + per-feature learnable log_sigma.

    Forward: mu = z @ W + b, log_sigma = log_sigma_param (broadcast to batch).
    """

    def __init__(
        self,
        in_dim: int,
        n_features: int,
        feature_prior: FeatureStructurePrior | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_features = n_features
        self.W = nn.Parameter(torch.randn(in_dim, n_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_features))
        self.log_sigma_param = nn.Parameter(torch.zeros(n_features))
        self.feature_prior = feature_prior or NoFeaturePrior()

    def loading_matrix(self) -> Tensor:
        return self.W

    def forward(
        self,
        z: Tensor,
        batch_cov: Tensor | None = None,
        size_factor: Tensor | None = None,  # ignored
    ) -> LikelihoodParams:
        mu = z @ self.W + self.b
        log_sigma = self.log_sigma_param.expand_as(mu)
        return LikelihoodParams(mu=mu, log_sigma=log_sigma)


class BernoulliDecoder(Decoder):
    """Linear loadings -> logits."""

    def __init__(
        self,
        in_dim: int,
        n_features: int,
        feature_prior: FeatureStructurePrior | None = None,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.n_features = n_features
        self.W = nn.Parameter(torch.randn(in_dim, n_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(n_features))
        self.feature_prior = feature_prior or NoFeaturePrior()

    def loading_matrix(self) -> Tensor:
        return self.W

    def forward(
        self,
        z: Tensor,
        batch_cov: Tensor | None = None,
        size_factor: Tensor | None = None,  # ignored
    ) -> LikelihoodParams:
        logits = z @ self.W + self.b
        return LikelihoodParams(mu=logits)
