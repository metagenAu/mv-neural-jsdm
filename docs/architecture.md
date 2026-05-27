# Architecture

`mv-neural-jsdm` is a multi-view VAE composed from interchangeable parts. The
forward path of the model is:

```
batch -> per-assay encoders -> Fusion -> joint (mu_z, logvar_z)
                                            |
                                            v
                                       HierarchyBlock
                                            |
                                            v
                                   (mu_q, logvar_q) = q(z)
                                            |
                                            v
                                  reparam -> z = mu_q + std * eps
                                            |
                          +-----------------+-----------------+
                          |                                   |
                  split shared / private                       |
                          |                                   |
                          v                                   v
                  per-assay Decoders   ...   per-assay Likelihoods (NLL)
```

## Block-level wiring

* **EnvCovariateEncoder** (skeleton when `env: none`). Takes the env covariates
  `env__*` and produces a conditioning vector concatenated with the encoder
  input and (optionally) the decoder input.

* **Per-assay Encoder** (smoke path: `CountEncoder` on log1p(x / sf)). Outputs
  a Gaussian factor `q_a(z) = N(mu_a, exp(logvar_a))` over the *joint* latent
  space (shared + private dims concatenated).

* **Fusion** combines the per-assay Gaussian factors into one joint
  Gaussian. `MoPoE` samples one PoE-subset of the present experts per
  minibatch; `PoE` uses all experts; `Concat` is a debug baseline.

* **HierarchyBlock** adds per-group offsets to the joint Gaussian. Each level
  is one of:
  * `hierarchical_prior` -- learnable per-group mean + a learnable per-level
    log-variance shared across groups. Proper partial pooling.
  * `embedding` -- per-group mean only, no variance contribution.
  * `off` -- contributes nothing.
  Multi-level contributions sum on the mean and on the variance.

* **Split** divides z into `z_shared` (first `shared_dim` components) and
  per-assay private blocks `z_private_a`.

* **Per-assay Decoder** (smoke path: `NBDecoder`) maps `[z_shared, z_private_a]`
  to likelihood parameters. The decoder owns its **loading matrix** `W` and a
  `FeatureStructurePrior` over that matrix.

* **Likelihood** computes the per-sample, masked NLL.

## ELBO

```
loss = sum_a w_a * recon_NLL_a
     + beta_shared * KL_shared
     + sum_a beta_private_a * KL_private_a
     + sum_a -log_prob(W_a)        # feature-structure prior, summed over rows
```

KL is closed-form against either `StandardNormalPrior` (no hierarchy) or
`HierarchicalPrior` with the hierarchy offsets as context.

## What is implemented vs. stubbed

Implemented end-to-end on the smoke path: data schema, loaders, transforms,
continuous-index scaler, splits, datamodule; CountEncoder, NBDecoder,
PoE/MoPoE/Concat, StandardNormalPrior, HierarchicalPrior, CompositePrior,
PhylogeneticPagel, PhylogeneticBrownian, NoFeaturePrior, HierarchyBlock
(all three modes), NBLikelihood, MVNeuralJSDM, the training loss, the
LightningModule, the Hydra CLI, and the integration smoke test.

`NotImplementedError` stubs: `ContinuousEncoder`, `BinaryEncoder`,
`ZINBDecoder`, `GaussianDecoder`, `BernoulliDecoder`, `NeuralARTransition`,
`GPLatentPrior`, `GraphLaplacian`, `TaxonomicGroupwise`, ZINB/Gaussian/
Bernoulli/Poisson likelihoods, all of `analysis/`, `interpret/`, `eval/`, and
the non-smoke experiment configs.

## Assay surface

The model can mix the following (kind, likelihood) combinations in a single
training run; per-assay encoders and decoders are dispatched in
`mvnjsdm._build_encoder` / `_build_decoder`:

| kind         | likelihood        | encoder            | decoder           | status |
|--------------|-------------------|--------------------|-------------------|--------|
| counts       | nb                | CountEncoder       | NBDecoder         | done   |
| counts       | zinb              | CountEncoder       | ZINBDecoder       | done   |
| continuous   | gaussian_masked   | ContinuousEncoder  | GaussianDecoder   | done   |
| binary       | bernoulli         | BinaryEncoder      | BernoulliDecoder  | done   |
| counts       | poisson           | CountEncoder       | (stub)            | pending |

ZINBDecoder mirrors NBDecoder and adds a per-feature dropout-logit head; the
`gate` is shared across units (a per-feature bias). The ZINB likelihood is
implemented in `models/likelihoods.ZINBLikelihood`.

## CLI surface

End-to-end use:

* `python -m mvnjsdm.cli.train +experiment=<name>` -- trains, writes
  `latents.parquet`, `config.yaml`, `model.ckpt`, `summary.json` under
  `output_dir`.
* `python -m mvnjsdm.cli.analyse run_dir=<output_dir> analyses=[...]` --
  reloads the run and writes per-analysis files into `<output_dir>/analysis/`.
* `python -m mvnjsdm.cli.export run_dir=<output_dir>` -- reloads the run and
  writes latents (parquet + csv + optional SEM-ready CSV).

The reload path lives in `cli/_reload.py` and uses the shared
`cli/_build.py:build_model_from_cfg` helper to rebuild the model from the
saved config and state dict.

## GP-indexed latents

Configured via `model.gp` (see `configs/gp/{off,phenological,spatial,phen_and_spatial}.yaml`).

For each indexed latent dim k a per-dim independent Gaussian Process is placed
over a continuous covariate vector x in R^d:

    z_k(x) ~ GP(0, K_theta(x, x'))

The covariate is sourced from `cont__*` columns referenced by
`model.gp.input_columns`. Kernels: `rbf`, `matern_3_2`, `matern_5_2`,
`periodic`. Hyperparameters (`log_lengthscale`, `log_outputscale`, optional
`log_period`) are learnable per indexed dim.

`model.gp.applies_to` chooses which slice of the joint latent the GP covers:
`shared` (default — covers the shared dims), `private` (covers all per-assay
private dims together), `all`, or an explicit list of dim indices.

Composition: the model's `CompositePrior` runs the hierarchical prior over
the *full* joint latent and the GP over its declared `dim_slice`. The two
contributions sum in the ELBO, so the GP acts as an *additional* prior
tightening on the indexed dims (rather than replacing the standard prior).

Caveats:

* Direct (non-sparse) GP. Builds and Cholesky-factorises the [B, B] kernel
  matrix per batch. Prefer batch sizes <= a few thousand for tractability.
  Sparse / inducing-point variants are deferred.
* Multi-input GPs are exposed as a single block whose kernel takes the
  concatenated input vector. For an explicit product kernel across e.g. a
  phenological and spatial coordinate, compose multiple GPLatentPriors at
  the Python level — the YAML surface only ships one block per model.
* GPyTorch is intentionally NOT used at runtime. We hand-roll the four
  kernel families above with `torch.cdist` to keep the dependency surface
  minimal. The GPyTorch entry in pyproject is left only as a documentation
  marker that we'd reach for it if/when inducing-point support lands.

## Neural-AR transition (skeleton-grade)

`model.ar.enabled = true` activates a small MLP transition

    z_{t+1} = z_t + MLP([z_t, delta_t])

scored under a Gaussian noise model with configurable `noise_scale`. Sorts
within each `group_level` (default: a single dummy group) by
`cont__<index_name>` and accumulates lag-1 log-prob into the ELBO. This is
crude: a proper state-space treatment (Kalman / particle smoothing of the
joint posterior) is left as future work.
