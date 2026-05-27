# Config matrix

The configs are split across orthogonal axes so that experiments can compose
presets without rewriting top-level files. The axes and presets:

| axis                | presets                                                                       |
|---------------------|-------------------------------------------------------------------------------|
| `assay`             | `counts_nb`, `counts_zinb_phylo`, `continuous_gaussian`, `binary_bernoulli`   |
| `env`               | `none`, `small`, `full`                                                       |
| `hierarchy`         | `flat`, `nested`, `nested_with_crossed_year`                                  |
| `gp`                | `off`, `phenological`, `spatial`, `phen_and_spatial`                          |
| `feature_structure` | `none`, `pagel_lambda`, `brownian`, `graph_laplacian`, `taxonomic_groupwise`  |
| `training`          | `smoke`, `default`                                                            |
| `data`              | `synthetic_smoke`, `bio_only_two_assays`, `soil_full`                         |
| `ar`                | set `model.ar.enabled=true` and `model.ar.index_name=cont__<name>` (no preset, inline only) |

## Experiment presets

| experiment                  | data                  | env    | hierarchy                    | training | status         |
|-----------------------------|-----------------------|--------|------------------------------|----------|----------------|
| `smoke`                     | `synthetic_smoke`     | none   | flat                         | smoke    | implemented    |
| `bio_only_residual`         | `bio_only_two_assays` | none   | flat                         | default  | data pending   |
| `bio_only_full`             | `bio_only_two_assays` | small  | nested                       | default  | data pending   |
| `soil_residual_flat`        | `soil_full`           | none   | flat                         | default  | data pending   |
| `soil_full_hierarchical`    | `soil_full`           | small  | nested                       | default  | data pending   |
| `soil_full_hierarchical_temporal`    | `soil_full`  | small  | nested + GP (`phenological`) | default  | data pending   |
| `soil_full_hierarchical_temporal_ar` | `soil_full`  | small  | nested + GP + AR             | default  | data pending   |
| `variance_partition`        | `soil_full`           | small  | nested                       | default  | data pending   |

The soil experiments compose four assays (`assay_a`, `assay_b` as NB counts,
`chem` as Gaussian, `indicators` as Bernoulli).

### Assay x likelihood axis

| assay kind   | available likelihoods                       |
|--------------|---------------------------------------------|
| counts       | `nb`, `zinb` (Poisson pending)              |
| continuous   | `gaussian_masked`                           |
| binary       | `bernoulli`                                 |

## Invocation examples

```bash
# Smoke test (default unless overridden)
uv run mvnjsdm-train +experiment=smoke

# Override a single axis
uv run mvnjsdm-train hierarchy=nested

# Compose explicitly
uv run mvnjsdm-train data=bio_only_two_assays training=default \
  hierarchy=nested feature_structure@feature_structures.assay_a=brownian
```

Note: `+experiment=...` uses Hydra's append form because experiment configs are
not in the default list of `config.yaml`.
