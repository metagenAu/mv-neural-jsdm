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

from .decoders import BernoulliDecoder, Decoder, GaussianDecoder, NBDecoder, ZINBDecoder
from .encoders import BinaryEncoder, ContinuousEncoder, CountEncoder, Encoder, EnvCovariateEncoder
from .feature_priors import (
    FeatureStructurePrior,
    GraphLaplacian,
    NoFeaturePrior,
    PhylogeneticPagel,
    TaxonomicGroupwise,
)
from .fusion import ExpertOutput, Fusion, build_fusion
from .hierarchy import HierarchyBlock, LevelSpec, empty_hierarchy
from .likelihoods import LIKELIHOODS, Likelihood, NBLikelihood
from .priors import (
    CompositePrior,
    GPLatentPrior,
    HierarchicalPrior,
    LatentPrior,
    StandardNormalPrior,
)


@dataclass
class AssaySpec:
    name: str
    kind: str
    likelihood: str
    n_features: int
    private_dim: int
    size_factor: bool
    tree_C: Any = None  # numpy array or None
    feature_structure: str = "none"  # 'none' | 'pagel' | 'brownian' | 'graph_laplacian' | 'taxonomic_groupwise'
    init_lambda: float = 0.5
    feature_structure_config: dict[str, Any] | None = None
    graph_L: Any = None  # numpy array or None
    taxonomy_groups: Any = None  # 1-D array of long or None


@dataclass
class GPSpec:
    enabled: bool = False
    n_dims: int = 0           # how many dims this GP covers
    dim_start: int = 0        # contiguous slice start
    input_columns: list[str] = None  # type: ignore[assignment]
    kernel: str = "rbf"
    init_lengthscale: float = 1.0
    init_outputscale: float = 1.0
    init_period: float | None = None
    jitter: float = 1e-4
    applies_to: str = "shared"  # 'shared' | 'private' | 'all' | list[int]


def _build_encoder(spec: AssaySpec, joint_dim: int, env_dim: int) -> Encoder:
    if spec.kind == "counts":
        return CountEncoder(spec.n_features, joint_dim, env_dim=env_dim)
    if spec.kind == "continuous":
        return ContinuousEncoder(spec.n_features, joint_dim, env_dim=env_dim)
    if spec.kind == "binary":
        return BinaryEncoder(spec.n_features, joint_dim, env_dim=env_dim)
    raise NotImplementedError(f"encoder for kind={spec.kind} not implemented")


def _build_decoder(spec: AssaySpec, in_dim: int) -> Decoder:
    fp: FeatureStructurePrior
    cfg = spec.feature_structure_config or {}
    init_alpha = float(cfg.get("init_alpha", 1.0))
    if spec.feature_structure == "pagel" and spec.tree_C is not None:
        fp = PhylogeneticPagel(spec.tree_C, init_lambda=spec.init_lambda)
    elif spec.feature_structure == "brownian" and spec.tree_C is not None:
        from .feature_priors import PhylogeneticBrownian

        fp = PhylogeneticBrownian(spec.tree_C)
    elif spec.feature_structure == "graph_laplacian" and spec.graph_L is not None:
        fp = GraphLaplacian(spec.graph_L, init_alpha=init_alpha)
    elif spec.feature_structure == "taxonomic_groupwise" and spec.taxonomy_groups is not None:
        fp = TaxonomicGroupwise(spec.taxonomy_groups, init_alpha=init_alpha)
    else:
        fp = NoFeaturePrior()
    if spec.likelihood == "nb":
        return NBDecoder(
            in_dim=in_dim,
            n_features=spec.n_features,
            feature_prior=fp,
            use_size_factor=spec.size_factor,
        )
    if spec.likelihood == "zinb":
        return ZINBDecoder(
            in_dim=in_dim,
            n_features=spec.n_features,
            feature_prior=fp,
            use_size_factor=spec.size_factor,
        )
    if spec.likelihood == "gaussian_masked":
        return GaussianDecoder(in_dim=in_dim, n_features=spec.n_features, feature_prior=fp)
    if spec.likelihood == "bernoulli":
        return BernoulliDecoder(in_dim=in_dim, n_features=spec.n_features, feature_prior=fp)
    raise NotImplementedError(f"decoder for likelihood={spec.likelihood} not implemented")


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
        gp_spec: GPSpec | None = None,
        ar_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.assay_specs = assay_specs
        self.shared_dim = shared_dim
        self.ar_enabled = ar_enabled
        self.gp_spec: GPSpec | None = gp_spec
        self.ar_config = ar_config or {}

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

        # priors (composite). Hierarchical covers all dims (legacy path); GP
        # block covers only its declared slice. Existing tests using the
        # hierarchical-only setup are unaffected because dim_slice=None
        # treats the prior as global.
        self.prior = self._build_prior()
        # Optional neural-AR transition block
        self.ar: Any = None
        if ar_enabled:
            from .ar import NeuralARTransition

            self.ar = NeuralARTransition(
                latent_dim=self.joint_dim,
                hidden=int(self.ar_config.get("hidden", 32)),
                noise_scale=float(self.ar_config.get("noise_scale", 1.0)),
            )

    # ----- prior construction -------------------------------------------
    def _resolve_gp_slice(self) -> tuple[int, int]:
        """Resolve the (start, stop) contiguous dim slice covered by the GP."""
        spec = self.gp_spec
        if spec is None:
            return (0, 0)
        applies = spec.applies_to
        if applies == "shared":
            return (0, self.shared_dim)
        if applies == "all":
            return (0, self.joint_dim)
        if applies == "private":
            # Cover the union of private dims (contiguous: starts at shared_dim).
            return (self.shared_dim, self.joint_dim)
        if isinstance(applies, (list, tuple)):
            if len(applies) == 0:
                return (0, 0)
            return (int(min(applies)), int(max(applies)) + 1)
        raise ValueError(f"unknown gp.applies_to={applies!r}")

    def _build_prior(self) -> CompositePrior:
        priors: list[LatentPrior] = [HierarchicalPrior()]
        slices: list[tuple[int, int] | None] = [None]
        if self.gp_spec is not None and self.gp_spec.enabled:
            a, b = self._resolve_gp_slice()
            n_dims = b - a
            if n_dims > 0:
                input_dim = max(1, len(self.gp_spec.input_columns or []))
                gp = GPLatentPrior(
                    n_dims=n_dims,
                    input_dim=input_dim,
                    kernel=self.gp_spec.kernel,
                    init_lengthscale=self.gp_spec.init_lengthscale,
                    init_outputscale=self.gp_spec.init_outputscale,
                    init_period=self.gp_spec.init_period,
                    jitter=self.gp_spec.jitter,
                )
                # Mutate stored spec so callers know the resolved slice
                self.gp_spec.dim_start = a
                self.gp_spec.n_dims = n_dims
                priors.append(gp)
                slices.append((a, b))
        return CompositePrior(priors, dim_slices=slices)

    @property
    def has_gp(self) -> bool:
        return self.gp_spec is not None and self.gp_spec.enabled and self.gp_spec.n_dims > 0

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

    def gp_indices_from_batch(self, batch: dict[str, Any]) -> torch.Tensor | None:
        """Extract the cont__* columns this model's GP block indexes on.

        Uses ``batch['continuous_indices']`` (a dict mapping cont col name to
        a [B] tensor) if provided, else falls back to slicing ``batch['cont']``
        via ``batch['cont_columns']``.
        """
        if not self.has_gp:
            return None
        cols = self.gp_spec.input_columns or []
        if not cols:
            return None
        # 1) preferred: explicit named dict
        cont_dict = batch.get("continuous_indices")
        if cont_dict is not None:
            picks = []
            for c in cols:
                if c not in cont_dict:
                    return None
                v = cont_dict[c]
                if v.dim() == 1:
                    v = v.unsqueeze(-1)
                picks.append(v)
            return torch.cat(picks, dim=-1)
        # 2) fallback: cont matrix + column-name list on the batch
        cont = batch.get("cont")
        names = batch.get("cont_columns")
        if cont is None or names is None:
            return None
        idx_map = {n: i for i, n in enumerate(names)}
        picks = []
        for c in cols:
            if c not in idx_map:
                return None
            picks.append(cont[:, idx_map[c]:idx_map[c] + 1])
        return torch.cat(picks, dim=-1)

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        env = batch.get("env", None)
        mu_z, lv_z, experts = self.encode(batch["assays"], env)
        mu_q, lv_q, mu_h, lv_h = self.combine_with_hierarchy(
            mu_z, lv_z, batch.get("group_ids", {})
        )
        z = self.reparameterise(mu_q, lv_q)
        dec = self.decode_all(z, batch["assays"])
        gp_indices = self.gp_indices_from_batch(batch)
        return {
            "z": z,
            "mu_q": mu_q,
            "logvar_q": lv_q,
            "mu_h": mu_h,
            "logvar_h": lv_h,
            "experts": experts,
            "decoded": dec,
            "gp_indices": gp_indices,
        }
