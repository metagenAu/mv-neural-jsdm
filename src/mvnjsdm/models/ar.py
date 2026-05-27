"""Neural AR transition on an ordered continuous index.

``ar.index_name`` in the config references a ``cont__<name>`` column. The
transition is

    z_{t+1} = z_t + MLP([z_t, delta_t]) * noise_scale

added as an extra prior term -0.5 * (z_{t+1} - f(z_t, delta))^2 / sigma^2 to
the ELBO. Documented limitation: this is a CRUDE first-pass AR; a proper
state-space treatment (Kalman / particle smoothing of the joint posterior) is
left as future work.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class NeuralARTransition(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 32, noise_scale: float = 1.0) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.noise_scale = float(noise_scale)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent_dim),
        )
        # zero-init final layer so the model starts as a trivial random walk
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def predict_next(self, z: Tensor, delta: Tensor) -> Tensor:
        """Predict z_{t+1} given z_t and the (positive) time-delta to t+1."""
        if delta.dim() == 1:
            delta = delta.unsqueeze(-1)
        inp = torch.cat([z, delta], dim=-1)
        return z + self.net(inp)

    def transition_log_prob(
        self,
        z: Tensor,
        time_index: Tensor,
        unit_groups: Tensor,
    ) -> Tensor:
        """Sum-of-log-prob of lag-1 transitions within each group.

        Within each group, sort by ``time_index`` and form (z_t, z_{t+1}) pairs.
        Returns a scalar.
        """
        if z.numel() == 0:
            return z.new_zeros(())
        B = z.shape[0]
        device = z.device
        sigma = self.noise_scale
        const = -0.5 * math.log(2.0 * math.pi) - math.log(sigma)
        total = z.new_zeros(())
        unique = unit_groups.unique()
        for g in unique:
            mask = unit_groups == g
            order = torch.arange(B, device=device)[mask]
            if order.numel() < 2:
                continue
            t_sub = time_index[order]
            sort_idx = torch.argsort(t_sub)
            order_sorted = order[sort_idx]
            z_sub = z[order_sorted]
            t_sorted = t_sub[sort_idx]
            z_t = z_sub[:-1]
            z_next = z_sub[1:]
            delta = (t_sorted[1:] - t_sorted[:-1]).clamp_min(1e-6)
            pred = self.predict_next(z_t, delta)
            resid = z_next - pred
            # log_prob per (transition, dim)
            lp = (const - 0.5 * (resid / sigma).pow(2)).sum()
            total = total + lp
        return total
