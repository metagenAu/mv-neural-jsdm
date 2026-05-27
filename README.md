# mv-neural-jsdm

A domain-agnostic **multi-view neural joint species distribution model** (jSDM).

Combines:
- Multi-view VAE with **MoPoE** fusion across heterogeneous assays
- **Hierarchical random effects** on the latent space (partial pooling)
- **Gaussian process priors** on latent dimensions indexed by continuous covariates (skeleton)
- Optional **neural auto-regressive** transition over an ordered continuous index (skeleton)
- **Feature-structure priors** on decoder loadings (Pagel's lambda, Brownian, graph Laplacian)
- Per-assay likelihoods: **NB / ZINB / Gaussian / Bernoulli / Poisson**

This is a **research codebase**. Correctness, configurability, and inspectability come
before performance. Domain meaning lives in configs and data, never in `src/`.

## Quickstart

```bash
# create env + install
make install

# smoke test (<3 min CPU)
make smoke

# train via Hydra
uv run mvnjsdm-train experiment=smoke
```

## Layout

```
src/mvnjsdm/
  data/      schema, loaders, transforms, feature_structures, datamodule
  models/    encoders, decoders, fusion, priors, feature_priors, hierarchy, ar, likelihoods, mvnjsdm
  training/  losses, lightning_module, callbacks
  analysis/  latent_export, variance_partition, interaction_matrix, ordination, coupling, sem_export
  interpret/ decoder_jacobian, feature_annotation_bridge, shap_decoder
  eval/      reconstruction, cross_assay, hmsc_crosscheck
  cli/       train, export, analyse
```

See `docs/architecture.md` for the wiring diagram and `docs/config_matrix.md` for
the preset matrix.

## Notebooks (planned)

- `01_explore_synthetic_smoke.ipynb`
- `02_inspect_latents.ipynb`
- `03_variance_partition.ipynb`
- `04_feature_prior_recovery.ipynb`

## Status

- Smoke path (NB counts, two assays, Pagel feature prior, flat hierarchy, no GP, no AR):
  fully implemented and tested.
- Continuous / Bernoulli encoders + decoders, GP prior, AR, analysis/interpret/eval
  modules: **skeleton with `NotImplementedError`** (see `docs/architecture.md`).
