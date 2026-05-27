"""Shared model-construction helpers used by both train and reload paths."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from ..data.datamodule import MVNeuralJSDMDataModule
from ..models.hierarchy import LevelSpec
from ..models.mvnjsdm import AssaySpec, GPSpec, MVNeuralJSDM
from ..training.lightning_module import MVNeuralJSDMLit
from ..training.losses import LossWeights


def build_assay_specs(cfg: DictConfig, dm: MVNeuralJSDMDataModule) -> list[AssaySpec]:
    specs: list[AssaySpec] = []
    private_dim_default = int(cfg.model.private_dim)
    shapes = dm.assay_shapes
    fs_cfgs = OmegaConf.to_container(cfg.model.get("feature_structures", {}), resolve=True) or {}
    for name, acfg in cfg.model.assays.items():
        d = OmegaConf.to_container(acfg, resolve=True)
        n_features = shapes[name]
        fs = fs_cfgs.get(name, {"kind": "none"})
        feature_structure = fs.get("kind", "none")
        init_lambda = float(fs.get("init_lambda", 0.5))
        tree_C = dm.get_tree_C(name) if feature_structure in ("pagel", "brownian") else None
        graph_L = dm.get_graph_L(name) if feature_structure == "graph_laplacian" else None
        tax_groups = (
            dm.get_taxonomy_groups(name)
            if feature_structure == "taxonomic_groupwise"
            else None
        )
        specs.append(
            AssaySpec(
                name=name,
                kind=d["kind"],
                likelihood=d["likelihood"],
                n_features=n_features,
                private_dim=int(d.get("private_dim", private_dim_default)),
                size_factor=bool(d.get("size_factor", False)),
                tree_C=tree_C,
                feature_structure=feature_structure,
                init_lambda=init_lambda,
                feature_structure_config=fs,
                graph_L=graph_L,
                taxonomy_groups=tax_groups,
            )
        )
    return specs


def build_gp_spec(cfg: DictConfig) -> GPSpec | None:
    gp_cfg = cfg.model.get("gp", None)
    if gp_cfg is None:
        return None
    d = OmegaConf.to_container(gp_cfg, resolve=True)
    if not d or not d.get("enabled", False):
        return GPSpec(enabled=False)
    # Resolve input columns: prefer 'input_columns', fall back to a single
    # 'index' string (legacy YAML shape).
    cols = d.get("input_columns")
    if cols is None:
        single = d.get("index")
        cols = [single] if single else []
    return GPSpec(
        enabled=True,
        input_columns=list(cols),
        kernel=str(d.get("kernel", "rbf")),
        init_lengthscale=float(d.get("init_lengthscale", 1.0)),
        init_outputscale=float(d.get("init_outputscale", 1.0)),
        init_period=(float(d["init_period"]) if d.get("init_period") is not None else None),
        jitter=float(d.get("jitter", 1e-4)),
        applies_to=d.get("applies_to", "shared"),
    )


def build_hierarchy_levels(cfg: DictConfig, dm: MVNeuralJSDMDataModule) -> list[LevelSpec]:
    h = cfg.model.get("hierarchy", None)
    if h is None:
        return []
    levels_cfg = OmegaConf.to_container(h.get("levels", []), resolve=True) or []
    specs: list[LevelSpec] = []
    for entry in levels_cfg:
        name = entry["name"]
        mode = entry.get("mode", "hierarchical_prior")
        if name not in dm.group_cardinality:
            continue
        specs.append(LevelSpec(name=name, n_groups=dm.group_cardinality[name], mode=mode))
    return specs


def build_datamodule(cfg: DictConfig) -> MVNeuralJSDMDataModule:
    seed = int(cfg.get("seed", 42))
    bs = int(cfg.training.get("batch_size", 32))
    size_factor_map: dict[str, bool] = {}
    for name, acfg in cfg.model.assays.items():
        size_factor_map[name] = bool(acfg.get("size_factor", False))
    dm = MVNeuralJSDMDataModule(
        data_root=cfg.data.data_root,
        batch_size=bs,
        val_frac=float(cfg.training.get("val_frac", 0.15)),
        seed=seed,
        assay_size_factor=size_factor_map,
    )
    dm.setup()
    return dm


def build_model_from_cfg(cfg: DictConfig, dm: MVNeuralJSDMDataModule) -> tuple[MVNeuralJSDM, MVNeuralJSDMLit]:
    """Return ``(MVNeuralJSDM, MVNeuralJSDMLit)`` built per ``cfg``.

    Datamodule is assumed already ``setup()``-ed.
    """
    assay_specs = build_assay_specs(cfg, dm)
    hierarchy_levels = build_hierarchy_levels(cfg, dm)
    env_dim = len(dm.env_cols) if cfg.model.get("env", {}).get("enabled", False) else 0
    gp_spec = build_gp_spec(cfg)
    ar_cfg = OmegaConf.to_container(cfg.model.get("ar", {}), resolve=True) or {}
    model = MVNeuralJSDM(
        assay_specs=assay_specs,
        shared_dim=int(cfg.model.shared_dim),
        fusion=str(cfg.model.fusion),
        hierarchy_levels=hierarchy_levels,
        env_dim=env_dim,
        ar_enabled=bool(ar_cfg.get("enabled", False)),
        gp_spec=gp_spec,
        ar_config=ar_cfg,
    )
    weights = LossWeights(
        beta_shared=float(cfg.model.beta_shared),
        beta_private={name: float(cfg.model.beta_private) for name in dm.assay_shapes},
    )
    lit = MVNeuralJSDMLit(model=model, weights=weights, lr=float(cfg.training.get("lr", 1e-3)))
    return model, lit
