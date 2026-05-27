"""Top-level multi-view neural jSDM module.

Composition:

    per-assay encoders -> Fusion -> joint (mu_z, logvar_z)
        |
        v  (precision combination with hierarchy offsets)
    HierarchyBlock random effects
        |
        v
    Split into z_shared (first ``shared_dim``) + z_private_a per assay
        |
        v
    Per-assay Decoder consuming [z_shared, z_private_a]
        |
        v
    Likelihood log_probs (masked)

The model returns an output dict suitable for loss assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .decoders import Decoder, NBDecoder
from .encoders import CountEncoder, Encoder, EnvCovariateEncoder
from .feature_priors import FeatureStructurePrior, NoFeaturePrior, PhylogeneticPagel
from .fusion import ExpertOutput, Fusion, build_fusion
from .hierarchy import HierarchyBlock, LevelSpec, empty_hierarchy
from .likelihoods import LIKELIHOODS, Likelihood, NBLikelihood
from .priors import CompositePrior, HierarchicalPrior, LatentPrior, StandardNormalPrior


@dataclass
class AssaySpec:
    name: str
    kind: str
    likelihood: str
    n_features: int
    private_dim: int
    size_factor: bool
    tree_C: Any = None  # numpy array or None
    feature_structure: str = "none"  # 'none' | 'pagel' | 'brownian'
    init_lambda: float = 0.5


def _build_encoder(spec: AssaySpec, joint_dim: int, env_dim: int) -> Encoder:
    if spec.kind == "counts":
        return CountEncoder(spec.n_features, joint_dim, env_dim=env_dim)
    raise NotImplementedError(f"encoder for kind={spec.kind} not in smoke path")


def _build_decoder(spec: AssaySpec, in_dim: int) -> Decoder:
    fp: FeatureStructurePrior
    if spec.feature_structure == "pagel" and spec.tree_C is not None:
        fp = PhylogeneticPagel(spec.tree_C, init_lambda=spec.init_lambda)
    elif spec.feature_structure == "brownian" and spec.tree_C is not None:
        from .feature_priors import PhylogeneticBrownian

        fp = PhylogeneticBrownian(spec.tree_C)
    else:
        fp = NoFeaturePrior()
    if spec.likelihood == "nb":
        return NBDecoder(
            in_dim=in_dim,
            n_features=spec.n_features,
            feature_prior=fp,
            use_size_factor=spec.size_factor,
        )
    raise NotImplementedError(f"decoder for likelihood={spec.likelihood} not in smoke path")


def _build_likelihood(spec: AssaySpec) -> Likelihood:
    cls = LIKELIHOODS[spec.likelihood]
    return cls()


class MVNeuralJSDM(nn.Module):
    def __init__(
        self,
        assay_specs: list[AssaySpec],
        shared_dim: int = 6,
        fusion: str = "mopoe",
        hierarchy_levels: list[LevelSpec] | None = None,
        env_dim: int = 0,
        env_hidden: int = 16,
        env_out_dim: int = 8,
        ar_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.assay_specs = assay_specs
        self.shared_dim = shared_dim
        self.ar_enabled = ar_enabled

        # private dims are stored as offsets into the joint vector
        self.private_dims = {s.name: s.private_dim for s in assay_specs}
        self.joint_dim = shared_dim + sum(self.private_dims.values())

        # offsets
        self.private_offsets: dict[str, int] = {}
        off = shared_dim
        for s in assay_specs:
            self.private_offsets[s.name] = off
            off += s.private_dim

        # env encoder
        self.env_enc = EnvCovariateEncoder(env_dim, hidden=env_hidden, out_dim=env_out_dim)

        # per-assay encoders: each produces (mu, logvar) of joint_dim
        self.encoders = nn.ModuleDict(
            {s.name: _build_encoder(s, self.joint_dim, self.env_enc.out_dim) for s in assay_specs}
        )
        self.fusion: Fusion = build_fusion(fusion)

        # decoders consume [z_shared, z_private_a]
        self.decoders = nn.ModuleDict(
            {s.name: _build_decoder(s, shared_dim + s.private_dim) for s in assay_specs}
        )
        self.likelihoods: dict[str, Likelihood] = {s.name: _build_likelihood(s) for s in assay_specs}

        # hierarchy
        if hierarchy_levels:
            self.hierarchy = HierarchyBlock(hierarchy_levels, self.joint_dim)
        else:
            self.hierarchy = empty_hierarchy(self.joint_dim)

        # priors
        self.prior = CompositePrior([HierarchicalPrior()])  # uses context offsets

    # ----- forward -------------------------------------------------------
    def encode(
        self,
        assays_batch: dict[str, dict[str, Tensor]],
        env: Tensor | None,
    ) -> tuple[Tensor, Tensor, dict[str, ExpertOutput]]:
        env_h = self.env_enc(env)
        experts: dict[str, ExpertOutput] = {}
        for s in self.assay_specs:
            data = assays_batch[s.name]
            x = data["x"]
            mask = data["mask"]
            sf = data.get("sf", None)
            # presence: a sample with all-zero mask is absent
            present = mask.sum(dim=-1) > 0
            mu, lv = self.encoders[s.name](x, mask, size_factor=sf, batch_cov=env_h)
            experts[s.name] = ExpertOutput(mu=mu, logvar=lv, present_mask=present)
        mu_z, lv_z = self.fusion(experts)
        return mu_z, lv_z, experts

    def combine_with_hierarchy(
        self,
        mu_z: Tensor,
        lv_z: Tensor,
        group_ids: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not getattr(self.hierarchy, "level_names", []):
            # no hierarchy configured: return mu_z, lv_z and zero offsets
            mu_h = torch.zeros_like(mu_z)
            lv_h = torch.full_like(lv_z, -50.0)
            return mu_z, lv_z, mu_h, lv_h
        mu_h, lv_h = self.hierarchy(group_ids)
        # Gaussian product (precisions add) -- treat hierarchy as a prior
        # actually here we add a (mean shift + extra variance) to q(z):
        # q' = N(mu_z + mu_h, var_z + var_h)
        # This matches the brief: "Reparam sample z = mu + sqrt(var + offset_var) * eps + offset_mu"
        mu_q = mu_z + mu_h
        var_q = lv_z.exp() + lv_h.exp()
        lv_q = torch.log(var_q.clamp_min(1e-8))
        return mu_q, lv_q, mu_h, lv_h

    def reparameterise(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_all(
        self,
        z: Tensor,
        assays_batch: dict[str, dict[str, Tensor]],
    ) -> dict[str, Any]:
        z_shared = z[:, : self.shared_dim]
        out: dict[str, Any] = {}
        for s in self.assay_specs:
            off = self.private_offsets[s.name]
            z_priv = z[:, off : off + s.private_dim]
            z_in = torch.cat([z_shared, z_priv], dim=-1)
            sf = assays_batch[s.name].get("sf", None)
            params = self.decoders[s.name](z_in, size_factor=sf)
            out[s.name] = params
        return out

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        env = batch.get("env", None)
        mu_z, lv_z, experts = self.encode(batch["assays"], env)
        mu_q, lv_q, mu_h, lv_h = self.combine_with_hierarchy(
            mu_z, lv_z, batch.get("group_ids", {})
        )
        z = self.reparameterise(mu_q, lv_q)
        dec = self.decode_all(z, batch["assays"])
        return {
            "z": z,
            "mu_q": mu_q,
            "logvar_q": lv_q,
            "mu_h": mu_h,
            "logvar_h": lv_h,
            "experts": experts,
            "decoded": dec,
        }
