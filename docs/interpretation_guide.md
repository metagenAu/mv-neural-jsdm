# Interpretation guide

This document collects pointers for inspecting a trained model.

## Latents

Latents are exported via `mvnjsdm.cli.train.run`, which writes
`outputs/<run>/latents.parquet` with columns `unit_id, split, z0, z1, ...`.
Re-load with pandas/pyarrow and join back on `unit_id` to inspect against
`units.parquet` covariates.

## Variance partitioning (skeleton)

`analysis/variance_partition.py` will eventually decompose total latent
variance into shared / private / hierarchy / GP / residual components. The
recipe: for each block of the latent and each hierarchy level, compute the
sample variance of its mean contribution and divide by the sample variance of
the full posterior mean.

## Feature priors

Each decoder's `feature_prior` exposes:

* `lambda_value()` (Pagel only) -- learned phylogenetic signal in `[0, 1]`.
* `sigma2()` -- learned scale.

Both are written to `summary.json` after training.

## Hierarchy

Each level's `nn.Embedding` weights give the per-group offset in z-space.
For levels in `hierarchical_prior` mode, the learned per-level `raw_logvar`
records the partial-pooling shrinkage strength.

## Decoder Jacobian (skeleton)

Backpropagating from a decoder output back to z gives the local sensitivity
of each feature to each latent dim. `interpret/decoder_jacobian.py` is a
skeleton.

## Latent collapse

`training.callbacks.LatentCollapseMonitor` logs the number of active latent
dimensions per validation epoch (variance of mu_q > 1e-2). Collapse indicates
over-regularisation (often `beta_*` too large) or insufficient signal.
