# Interpretation guide

This document collects pointers for inspecting a trained model.

## Pipeline at a glance

```
mvnjsdm-train +experiment=<name>  ->  outputs/<stamp>/
                                          model.ckpt
                                          config.yaml
                                          latents.parquet
                                          summary.json

mvnjsdm-analyse run_dir=outputs/<stamp>
    analyses=[variance_partition,latent_interaction,cross_assay,
              within_assay,ordination]
  ->  outputs/<stamp>/analysis/
          variance_partition.csv
          latent_interaction.csv
          cross_assay_<a>__<b>.parquet
          within_assay_<a>.parquet
          ordination.csv

mvnjsdm-export run_dir=outputs/<stamp> formats=[parquet,csv] sem_export=true
  ->  outputs/<stamp>/latents.parquet
      outputs/<stamp>/latents.csv
      outputs/<stamp>/latents_sem.csv   (R-friendly column names)
```

The reload path in `cli/_reload.py` rebuilds the LightningModule and
datamodule from the saved config + state dict; you can also drive
`analysis/*` and `interpret/*` programmatically against
`(lit, datamodule)` if you prefer notebooks.

## Latents

Latents are exported via `mvnjsdm.cli.train.run`, which writes
`outputs/<run>/latents.parquet` with columns `unit_id, split, z0, z1, ...`.
Re-load with pandas/pyarrow and join back on `unit_id` to inspect against
`units.parquet` covariates.

## Variance partitioning

`analysis/variance_partition.variance_partition(model, dm, levels=..., env_covariates=...)`
returns a DataFrame indexed by latent dim with one column per supplied factor
(plus `residual`). The default method is sequential (Type-I) SS regression
against one-hot encoded `group__*` columns and standardised `env__*` columns
detected on the datamodule's `units` frame. See the docstring for the `anova`
and `ablation` alternatives.

## Feature priors

Each decoder's `feature_prior` exposes:

* `lambda_value()` (Pagel only) -- learned phylogenetic signal in `[0, 1]`.
* `sigma2()` -- learned scale.

Both are written to `summary.json` after training.

## Hierarchy

Each level's `nn.Embedding` weights give the per-group offset in z-space.
For levels in `hierarchical_prior` mode, the learned per-level `raw_logvar`
records the partial-pooling shrinkage strength.

## Decoder Jacobian

`interpret/decoder_jacobian.decoder_jacobian(model, assay=..., z_ref=..., size_factor=...)`
returns the local `[K_in, F]` sensitivity matrix of `E[x_assay]` to the
decoder input z (linearisation around `z_ref`). Bernoulli decoders are mapped
through the sigmoid first so the result is in probability space; NB decoders
return `d rate / dz`; Gaussian decoders return `dmu / dz`.

`decoder_jacobian_dataset` returns the per-unit Jacobians stacked along axis 0
for downstream averaging or weighting.

## Latent collapse

`training.callbacks.LatentCollapseMonitor` logs the number of active latent
dimensions per validation epoch (variance of mu_q > 1e-2). Collapse indicates
over-regularisation (often `beta_*` too large) or insufficient signal.
