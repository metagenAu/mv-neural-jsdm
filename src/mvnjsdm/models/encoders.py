"""Per-assay encoders. Smoke path implements a CountEncoder; others are stubs."""

from __future__ import annotations

import abc

import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module, abc.ABC):
    """Abstract encoder interface.

    Returns (mu_z, logvar_z) of shape [B, K_joint] where K_joint includes
    shared + (this assay's) private dims; the model concatenates these across
    assays during fusion.
    """

    @abc.abstractmethod
    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        size_factor: Tensor | None = None,
        batch_cov: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]: ...


class CountEncoder(Encoder):
    """MLP encoder on log1p(x / size_factor)."""

    def __init__(
        self,
        n_features: int,
        out_dim: int,
        hidden: int = 64,
        depth: int = 2,
        env_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = n_features + env_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.head_mu = nn.Linear(d, out_dim)
        self.head_lv = nn.Linear(d, out_dim)
        self.env_dim = env_dim

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        size_factor: Tensor | None = None,
        batch_cov: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if size_factor is not None:
            sf = size_factor.clamp_min(1.0).unsqueeze(-1)
            xn = torch.log1p(x / sf)
        else:
            xn = torch.log1p(x)
        xn = xn * mask
        if batch_cov is not None and self.env_dim > 0:
            xn = torch.cat([xn, batch_cov.float()], dim=-1)
        h = self.body(xn)
        mu = self.head_mu(h)
        lv = self.head_lv(h).clamp(min=-8.0, max=8.0)
        return mu, lv


class ContinuousEncoder(Encoder):
    """MLP encoder for continuous features (assumed pre-scaled).

    The input is multiplied by the per-feature mask (so missing features
    contribute zero) and the mask is concatenated as an extra channel to give
    the encoder a signal that the feature is absent.
    """

    def __init__(
        self,
        n_features: int,
        out_dim: int,
        hidden: int = 64,
        depth: int = 2,
        env_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = n_features * 2 + env_dim  # x + mask concatenated
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.head_mu = nn.Linear(d, out_dim)
        self.head_lv = nn.Linear(d, out_dim)
        self.env_dim = env_dim
        self.n_features = n_features

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        size_factor: Tensor | None = None,  # ignored
        batch_cov: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        xn = x * mask
        h_in = torch.cat([xn, mask], dim=-1)
        if batch_cov is not None and self.env_dim > 0:
            h_in = torch.cat([h_in, batch_cov.float()], dim=-1)
        h = self.body(h_in)
        mu = self.head_mu(h)
        lv = self.head_lv(h).clamp(min=-8.0, max=8.0)
        return mu, lv


class BinaryEncoder(Encoder):
    """MLP encoder for binary features in {0,1}; NaN inputs assumed pre-replaced
    with 0 and recorded in the mask.
    """

    def __init__(
        self,
        n_features: int,
        out_dim: int,
        hidden: int = 64,
        depth: int = 2,
        env_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = n_features * 2 + env_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.body = nn.Sequential(*layers) if layers else nn.Identity()
        self.head_mu = nn.Linear(d, out_dim)
        self.head_lv = nn.Linear(d, out_dim)
        self.env_dim = env_dim
        self.n_features = n_features

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        size_factor: Tensor | None = None,  # ignored
        batch_cov: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        xn = x * mask
        h_in = torch.cat([xn, mask], dim=-1)
        if batch_cov is not None and self.env_dim > 0:
            h_in = torch.cat([h_in, batch_cov.float()], dim=-1)
        h = self.body(h_in)
        mu = self.head_mu(h)
        lv = self.head_lv(h).clamp(min=-8.0, max=8.0)
        return mu, lv


class EnvCovariateEncoder(nn.Module):
    """Returns a conditioning vector, or None if env is None."""

    def __init__(self, env_dim: int, hidden: int = 16, out_dim: int = 8) -> None:
        super().__init__()
        self.env_dim = env_dim
        if env_dim == 0:
            self.net = None
            self.out_dim = 0
        else:
            self.net = nn.Sequential(
                nn.Linear(env_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, out_dim),
            )
            self.out_dim = out_dim

    def forward(self, e: Tensor | None) -> Tensor | None:
        if e is None or self.net is None:
            return None
        return self.net(e)
