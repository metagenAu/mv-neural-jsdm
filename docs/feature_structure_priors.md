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

## Graph Laplacian

`GraphLaplacian` implements a Tikhonov-style smoothness regulariser over a
feature graph (e.g. KEGG, taxonomy). Given the combinatorial Laplacian
`L = D - A` of the feature graph, the log-prob is

```
log p(W) = sum_k -alpha * W[k, :] @ L @ W[k, :]
```

with a learnable positive scalar `alpha` (softplus reparameterisation). This is
a regulariser, not a proper density: when the graph is connected `L` has a
zero eigenvalue from the constant vector, so the normalising constant is
dropped. The penalty grows with the total variation of each row of `W` along
graph edges, so neighbour-on-graph features tend to receive similar loadings.

Interpretation:
* `alpha` near 0: the prior is effectively inactive.
* `alpha` large: row-wise loadings become piecewise-smooth across the graph.

## Taxonomic groupwise

`TaxonomicGroupwise` implements an l_{2,1}-style group-sparsity prior. Given
`group_assignment[f] = g` mapping each feature to a taxonomic group id, the
log-prob is

```
log p(W) = -alpha * sum_k sum_g ||W[k, group == g]||_2
```

Each latent dim pays a cost proportional to the number of taxonomic groups it
uses (counted in l_2 norm rather than cardinality), encouraging each dim to
concentrate weight in a few groups rather than spreading uniformly. `alpha` is
learnable via softplus.

Interpretation:
* `alpha` near 0: prior inactive; loadings spread freely across groups.
* `alpha` large: loadings become block-sparse -- each latent dim "claims" a
  small number of groups.

## Interpretation

* Learned `lambda` near 0: features behave independently in the latent
  loadings (no useful phylogenetic signal).
* Learned `lambda` near 1: loadings track tree structure tightly.
* `sigma^2`: overall scale of the loadings; if it drifts very small, the
  decoder is leaning on the bias term and the latent is contributing little.
