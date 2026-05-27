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
