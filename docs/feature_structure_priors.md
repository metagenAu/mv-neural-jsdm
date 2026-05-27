# Feature-structure priors

The decoder loading matrix `W` has shape `[K, F]` (K = latent dim used by the
decoder, F = number of features in the assay). Each row `W[k, :]` is the
"trait vector" of latent dim k across features. A feature-structure prior puts
a multivariate-Gaussian prior on each row, decoupled across rows.

## Pagel's lambda (row-wise)

`PhylogeneticPagel` takes a tree variance-covariance matrix `C` (shape
`[F, F]`) computed as the shared root-to-MRCA path length between feature
pairs (see `data/feature_structures.py`). The per-row prior is

```
W[k, :] ~ N(0, sigma^2 (lambda * C + (1 - lambda) * I))
```

with `sigma^2 > 0` (softplus reparameterisation) and `lambda in [0, 1]`
(sigmoid reparameterisation). Both are learnable. `lambda = 0` recovers the
star-tree (independent features) and `lambda = 1` recovers full Brownian
motion.

Each forward pass computes a Cholesky of `Sigma = sigma^2 (lambda C +
(1 - lambda) I)` once and reuses it for the K rows via triangular solve.

## Brownian (lambda = 1 special case)

`PhylogeneticBrownian` removes the `(1 - lambda) I` shrinkage:

```
W[k, :] ~ N(0, sigma^2 * C).
```

Numerically less stable when `C` is near-singular; a small jitter is added.

## Graph Laplacian (skeleton)

Interpretation: a prior of the form `W[k, :] ~ N(0, sigma^2 (L + tau I)^{-1})`
encourages smoothness over a feature graph (e.g. KEGG, taxonomy). `L` is the
combinatorial Laplacian `D - A`. The skeleton stores the Laplacian; the
log-prob is not yet implemented.

## Taxonomic groupwise (skeleton)

Block-structured prior: features grouped at one taxonomic level share a
group-level mean. Implementation pending.

## Interpretation

* Learned `lambda` near 0: features behave independently in the latent
  loadings (no useful phylogenetic signal).
* Learned `lambda` near 1: loadings track tree structure tightly.
* `sigma^2`: overall scale of the loadings; if it drifts very small, the
  decoder is leaning on the bias term and the latent is contributing little.
