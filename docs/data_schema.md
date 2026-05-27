# Data schema

A dataset is a directory containing:

* `units.parquet` -- one row per unit. Required column: `unit_id`. Optional
  prefix-based columns:
  * `group__<level>` -- categorical grouping factor.
  * `env__<name>` -- numeric environmental covariate.
  * `cont__<name>` -- continuous index (e.g. time, position).
  * `batch__<name>` -- categorical batch/technical covariate.

* `assay__<name>__counts.parquet` -- long-form counts table with columns
  `unit_id`, `feature_id`, `count`.

* `assay__<name>__values.parquet` -- wide table with `unit_id` + feature
  columns (continuous assays).

* `assay__<name>__binary.parquet` -- wide table with `unit_id` + feature
  columns (binary assays).

* `assay__<name>__features.parquet` (optional) -- `feature_id` + annotation
  columns. Provides the canonical feature ordering.

* `assay__<name>__tree.newick` (optional) -- Newick phylogenetic tree.

* `assay__<name>__graph.edges.parquet` (optional) -- edge list with `source`,
  `target` (and optional weight) for graph-Laplacian priors.

* `assay__<name>__taxonomy.parquet` (optional) -- taxonomy table for
  group-wise priors.

The loader (`mvnjsdm.data.loaders.load_dataset`) returns a `MuData` object
whose modalities mirror the assays. Unit metadata is copied into each
modality's `obs`; the tree covariance and graph Laplacian, when available,
are placed in `varm["tree_C"]` / `varm["graph_L"]`.

## Domain agnosticism

The schema does not bake in any domain nouns. The prefix conventions
(`group__`, `env__`, `cont__`, `batch__`, `assay__<name>__`) are the only
contracts; everything else is data.
