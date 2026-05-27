"""Latent priors over the joint z space.

All priors expose:

    log_prob(z, context) -> [B]
    kl_divergence(mu, logvar, context) -> [B]

Closed-form KL is used where available.

The :class:`CompositePrior` supports a per-prior ``dim_slice`` (a contiguous
range of latent dims that prior applies to). Sub-priors that don't set a
``dim_slice`` are treated as applying to *all* dims of the input (back-compat
with the previous global priors). When two priors cover overlapping dims their
log_probs / KLs are summed — for the GP-vs-standard composition we therefore
arrange the GP to cover a sub-slice and the standard-normal to cover the
*complement*.
"""

from __future__ import annotations

import abc
import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._kernels import get_kernel


class LatentPrior(nn.Module, abc.ABC):
    @abc.abstractmethod
    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor: ...

    @abc.abstractmethod
    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor: ...


class StandardNormalPrior(LatentPrior):
    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor:
        return (-0.5 * (z ** 2 + math.log(2 * math.pi))).sum(dim=-1)

    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor:
        return 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)


class HierarchicalPrior(LatentPrior):
    """Random-effect prior whose mean is shifted by hierarchy offsets.

    Context expected to contain ``offset_mu`` and ``offset_logvar`` from a
    HierarchyBlock; the prior becomes N(offset_mu, exp(offset_logvar)).
    """

    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor:
        if context is None or "offset_mu" not in context:
            mu_p = torch.zeros_like(z)
            lv_p = torch.zeros_like(z)
        else:
            mu_p = context["offset_mu"]
            lv_p = context["offset_logvar"]
        var_p = lv_p.exp()
        return (-0.5 * ((z - mu_p) ** 2 / var_p + lv_p + math.log(2 * math.pi))).sum(dim=-1)

    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor:
        if context is None or "offset_mu" not in context:
            mu_p = torch.zeros_like(mu)
            lv_p = torch.zeros_like(logvar)
        else:
            mu_p = context["offset_mu"]
            lv_p = context["offset_logvar"]
        var_p = lv_p.exp()
        var_q = logvar.exp()
        kl = 0.5 * (
            (var_q + (mu - mu_p).pow(2)) / var_p
            - 1.0
            + lv_p
            - logvar
        )
        return kl.sum(dim=-1)


class GPLatentPrior(LatentPrior):
    """GP prior on selected latent dims indexed by continuous covariates.

    Per-dim independent GPs: each indexed latent dim k follows

        z_k(x) ~ GP(0, K_theta(x, x'))

    with its own (lengthscale, outputscale, [period]). Indices flow in via
    ``context['indices']`` of shape ``[B, d]``.

    Direct (non-sparse) GP: builds the ``[B, B]`` kernel matrix once per call
    and reuses the Cholesky across all indexed dims. Document caveat: prefer
    sparse / inducing-point methods if B exceeds a few thousand.
    """

    def __init__(
        self,
        n_dims: int,
        input_dim: int,
        kernel: str = "rbf",
        init_lengthscale: float = 1.0,
        init_outputscale: float = 1.0,
        init_period: float | None = None,
        jitter: float = 1e-4,
    ) -> None:
        super().__init__()
        if n_dims <= 0:
            raise ValueError("n_dims must be positive")
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.n_dims = int(n_dims)
        self.input_dim = int(input_dim)
        self.kernel_name = kernel
        self._kernel_fn = get_kernel(kernel)
        self.jitter = float(jitter)

        # Per-dim ARD-style hyperparameters: shape [n_dims, input_dim] (lengthscale),
        # [n_dims] (outputscale). Stored in log space and softplused on use.
        ls_init = max(float(init_lengthscale), 1e-3)
        os_init = max(float(init_outputscale), 1e-3)
        self.raw_log_lengthscale = nn.Parameter(
            torch.full((self.n_dims, self.input_dim), math.log(ls_init))
        )
        self.raw_log_outputscale = nn.Parameter(
            torch.full((self.n_dims,), math.log(os_init))
        )
        if kernel == "periodic":
            p_init = float(init_period) if init_period is not None else 1.0
            self.raw_log_period = nn.Parameter(
                torch.full((self.n_dims, self.input_dim), math.log(max(p_init, 1e-3)))
            )
        else:
            self.register_parameter("raw_log_period", None)

    # ------------------------------------------------------------------ utils
    def lengthscale(self, dim_index: int) -> Tensor:
        return F.softplus(self.raw_log_lengthscale[dim_index]) + 1e-6

    def outputscale(self, dim_index: int) -> Tensor:
        return F.softplus(self.raw_log_outputscale[dim_index]) + 1e-6

    def period(self, dim_index: int) -> Tensor:
        if self.raw_log_period is None:
            raise RuntimeError("kernel has no period parameter")
        return F.softplus(self.raw_log_period[dim_index]) + 1e-6

    def _kernel_matrix(self, x: Tensor, dim_index: int) -> Tensor:
        ls = self.lengthscale(dim_index)
        os_ = self.outputscale(dim_index)
        if self.kernel_name == "periodic":
            return self._kernel_fn(x, x, ls, os_, self.period(dim_index))
        return self._kernel_fn(x, x, ls, os_)

    def _indices_from_context(self, context: dict | None, B: int, device, dtype) -> Tensor:
        if context is None or "indices" not in context or context["indices"] is None:
            raise ValueError(
                "GPLatentPrior requires context['indices'] of shape [B, d]"
            )
        idx = context["indices"]
        if idx.dim() == 1:
            idx = idx.unsqueeze(-1)
        if idx.shape[0] != B:
            raise ValueError(
                f"GP indices batch ({idx.shape[0]}) != z batch ({B})"
            )
        if idx.shape[1] != self.input_dim:
            raise ValueError(
                f"GP indices have input_dim={idx.shape[1]}, expected {self.input_dim}"
            )
        return idx.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ API
    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor:
        """Sum-over-dims, *per-sample-averaged* log-prob.

        Returns a tensor of shape ``[B]`` for compositional purposes: each
        element is ``total_log_prob / B`` so it summates correctly when the
        loss takes ``mean(dim=0)``.
        """
        if z.shape[-1] != self.n_dims:
            raise ValueError(
                f"GPLatentPrior expected z with last dim {self.n_dims}, got {z.shape[-1]}"
            )
        B = z.shape[0]
        x = self._indices_from_context(context, B, z.device, z.dtype)

        total = z.new_zeros(())
        const = -0.5 * B * math.log(2.0 * math.pi)
        for k in range(self.n_dims):
            K = self._kernel_matrix(x, k)
            K = K + self.jitter * torch.eye(B, device=z.device, dtype=z.dtype)
            L = torch.linalg.cholesky(K)
            zk = z[:, k]
            y = torch.linalg.solve_triangular(L, zk.unsqueeze(-1), upper=False).squeeze(-1)
            quad = (y * y).sum()
            logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
            total = total + (const - 0.5 * logdet - 0.5 * quad)
        # Broadcast scalar back to [B] (per-sample mean form)
        return (total / max(B, 1)).expand(B)

    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor:
        """KL[q(z) || p_GP(z|x)] for mean-field q.

        Closed-ish form: with q = N(mu, diag(var)),

            KL = -E_q[log p_GP(z)] - H[q]

        and E_q[log p_GP(z)] = log p_GP(mu) - 0.5 * sum_k tr(K_k^{-1} diag(var_k))
        (each dim k has its own ``K_k``).

        Returns shape ``[B]`` (KL_total / B per element) so it composes with
        ``.mean(dim=0)`` in the loss.
        """
        if mu.shape != logvar.shape:
            raise ValueError("mu/logvar shape mismatch")
        if mu.shape[-1] != self.n_dims:
            raise ValueError(
                f"GPLatentPrior expected last dim {self.n_dims}, got {mu.shape[-1]}"
            )
        B = mu.shape[0]
        x = self._indices_from_context(context, B, mu.device, mu.dtype)
        var = logvar.exp()

        # Entropy of mean-field Gaussian: H = 0.5 sum (log(2pi e) + logvar)
        entropy = 0.5 * (math.log(2.0 * math.pi * math.e) * self.n_dims * B + logvar.sum())

        # E_q[log p_GP]
        total_elogp = mu.new_zeros(())
        const = -0.5 * B * math.log(2.0 * math.pi)
        for k in range(self.n_dims):
            K = self._kernel_matrix(x, k)
            K = K + self.jitter * torch.eye(B, device=mu.device, dtype=mu.dtype)
            L = torch.linalg.cholesky(K)
            muk = mu[:, k]
            y = torch.linalg.solve_triangular(L, muk.unsqueeze(-1), upper=False).squeeze(-1)
            quad_mu = (y * y).sum()
            logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
            logp_mu = const - 0.5 * logdet - 0.5 * quad_mu
            # tr(K^{-1} diag(var_k)) = sum_b var_k[b] * (K^{-1})_{bb}
            # Compute diag(K^{-1}) via solving L L^T A = I -> A = K^{-1}
            # Cheaper: solve L L^T v = e_b for each b? Use a single solve.
            eye_B = torch.eye(B, device=mu.device, dtype=mu.dtype)
            Kinv = torch.cholesky_solve(eye_B, L)
            diag_Kinv = torch.diagonal(Kinv)
            trace_term = (var[:, k] * diag_Kinv).sum()
            total_elogp = total_elogp + (logp_mu - 0.5 * trace_term)

        kl_total = -total_elogp - entropy
        return (kl_total / max(B, 1)).expand(B)


class CompositePrior(LatentPrior):
    """Sum of constituent priors' log-probs / KLs, optionally over dim slices.

    Each constituent prior may declare a contiguous ``dim_slice`` (a tuple
    ``(start, stop)``); the composite will pass only that slice of z/mu/logvar
    to that constituent. Priors without a registered slice are evaluated on
    the full tensor (legacy behaviour).
    """

    def __init__(
        self,
        priors: Iterable[LatentPrior],
        dim_slices: list[tuple[int, int] | None] | None = None,
    ) -> None:
        super().__init__()
        priors = list(priors)
        self.priors = nn.ModuleList(priors)
        if dim_slices is None:
            dim_slices = [None] * len(priors)
        if len(dim_slices) != len(priors):
            raise ValueError("dim_slices must align with priors")
        self.dim_slices: list[tuple[int, int] | None] = list(dim_slices)

    def _slice(self, t: Tensor, sl: tuple[int, int] | None) -> Tensor:
        if sl is None:
            return t
        a, b = sl
        return t[..., a:b]

    def log_prob(self, z: Tensor, context: dict | None = None) -> Tensor:
        terms = []
        for p, sl in zip(self.priors, self.dim_slices):
            terms.append(p.log_prob(self._slice(z, sl), context))
        return torch.stack(terms, dim=0).sum(dim=0)

    def kl_divergence(
        self, mu: Tensor, logvar: Tensor, context: dict | None = None
    ) -> Tensor:
        terms = []
        for p, sl in zip(self.priors, self.dim_slices):
            if sl is None:
                mu_s, lv_s, ctx = mu, logvar, context
            else:
                a, b = sl
                mu_s = mu[..., a:b]
                lv_s = logvar[..., a:b]
                # Also slice offset_mu / offset_logvar if present.
                if context is not None and "offset_mu" in context:
                    ctx = dict(context)
                    ctx["offset_mu"] = context["offset_mu"][..., a:b]
                    ctx["offset_logvar"] = context["offset_logvar"][..., a:b]
                else:
                    ctx = context
            terms.append(p.kl_divergence(mu_s, lv_s, ctx))
        return torch.stack(terms, dim=0).sum(dim=0)
