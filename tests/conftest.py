"""Pytest fixtures including the synthetic smoke dataset generator.

Generates two NB-counted assays nested in 4 groups, with random binary trees
defining per-assay phylogenetic correlation. Writes a dataset to
``data/examples/synthetic_smoke`` (gitignored) on first run and emits
``truth.json`` next to it.

Used by tests/integration/test_smoke_end_to_end.py and by direct training runs.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SYN_DIR = REPO_ROOT / "data" / "examples" / "synthetic_smoke"


# ---------------------------------------------------------------------------
# Newick + covariance helpers (small, self-contained for the fixture)
# ---------------------------------------------------------------------------


def _random_binary_tree(n_leaves: int, rng: random.Random) -> tuple[str, np.ndarray, list[str]]:
    """Return (newick, C, leaf_names) for a random binary tree with branch lens.

    Branch lengths drawn uniform(0.1, 1.0). Internal nodes unnamed.
    """
    leaves = [f"L{i}" for i in range(n_leaves)]

    # Each "node" is a dict: {newick: str, subtree_leaves: list[str], distance_from_subtree_root: dict[leaf -> root]}
    # We build by repeatedly joining two random nodes.
    nodes: list[dict] = []
    for name in leaves:
        nodes.append(
            {
                "newick": name,
                "leaves": [name],
                "dist": {name: 0.0},  # distance from this subtree's root to its leaves
            }
        )

    while len(nodes) > 1:
        i, j = rng.sample(range(len(nodes)), 2)
        a = nodes.pop(max(i, j))
        b = nodes.pop(min(i, j))
        len_a = rng.uniform(0.1, 1.0)
        len_b = rng.uniform(0.1, 1.0)
        merged = {
            "newick": f"({a['newick']}:{len_a:.4f},{b['newick']}:{len_b:.4f})",
            "leaves": a["leaves"] + b["leaves"],
            "dist": {},
        }
        for leaf, d in a["dist"].items():
            merged["dist"][leaf] = d + len_a
        for leaf, d in b["dist"].items():
            merged["dist"][leaf] = d + len_b
        nodes.append(merged)

    root = nodes[0]
    newick = root["newick"] + ";"

    # Compute C: shared-path-from-root for each pair.
    # Equivalent: C[i,j] = root_depth[i] - distance_to_mrca(i,j)/2? Simpler approach:
    # build via reconstructing the tree's parent pointers from newick by re-parsing.
    from mvnjsdm.data.feature_structures import tree_variance_covariance

    names, C = tree_variance_covariance(newick)
    # Reorder C to match the leaf naming order we want (L0..L_{n-1}).
    # Our newick uses the same labels; just permute.
    idx = {n: i for i, n in enumerate(names)}
    order = [idx[name] for name in leaves]
    C = C[np.ix_(order, order)]
    return newick, C, leaves


def _draw_loadings(K: int, F: int, C: np.ndarray, lam: float, sigma2: float, rng: np.random.Generator) -> np.ndarray:
    Sigma = sigma2 * (lam * C + (1 - lam) * np.eye(F))
    L = np.linalg.cholesky(Sigma + 1e-6 * np.eye(F))
    W = np.zeros((K, F), dtype=np.float64)
    for k in range(K):
        eps = rng.standard_normal(F)
        W[k] = L @ eps
    return W


def _generate_synthetic_smoke(out: Path) -> dict:
    """Build the synthetic smoke dataset. Idempotent: re-uses existing if valid."""
    truth_path = out / "truth.json"
    units_path = out / "units.parquet"
    if truth_path.exists() and units_path.exists():
        try:
            return json.loads(truth_path.read_text())
        except Exception:
            pass

    out.mkdir(parents=True, exist_ok=True)

    py_rng = random.Random(0)
    rng = np.random.default_rng(0)

    n_units = 120
    n_groups = 4
    n_features_a = 50
    n_features_b = 50
    K = 6
    lam_true = {"assay_a": 0.7, "assay_b": 0.3}
    sigma2_true = 0.5

    # Trees + covariance
    newick_a, C_a, leaves_a = _random_binary_tree(n_features_a, py_rng)
    newick_b, C_b, leaves_b = _random_binary_tree(n_features_b, py_rng)

    (out / "assay__assay_a__tree.newick").write_text(newick_a)
    (out / "assay__assay_b__tree.newick").write_text(newick_b)

    # Draw true loadings
    W_a = _draw_loadings(K, n_features_a, C_a, lam_true["assay_a"], sigma2_true, rng)
    W_b = _draw_loadings(K, n_features_b, C_b, lam_true["assay_b"], sigma2_true, rng)
    intercept_a = rng.uniform(-1.0, 1.0, size=n_features_a)
    intercept_b = rng.uniform(-1.0, 1.0, size=n_features_b)

    # Group ids and group-level latent offsets
    group_labels = [f"G{i}" for i in range(n_groups)]
    unit_group = [py_rng.choice(group_labels) for _ in range(n_units)]
    group_offset = {g: rng.normal(0, 0.5, size=K) for g in group_labels}
    z = np.stack(
        [rng.standard_normal(K) + group_offset[unit_group[i]] for i in range(n_units)], axis=0
    )

    # Counts: NB with rate = exp(z @ W + b) * sf
    sf_a = rng.uniform(800, 1200, size=n_units)
    sf_b = rng.uniform(800, 1200, size=n_units)
    theta_a = 5.0
    theta_b = 5.0

    def sample_nb(rate: np.ndarray, theta: float) -> np.ndarray:
        # NB via Gamma-Poisson
        # shape=theta, scale=mu/theta
        g = rng.gamma(theta, rate / theta + 1e-9)
        return rng.poisson(g)

    log_rate_a = z @ W_a + intercept_a
    log_rate_a = np.clip(log_rate_a, -10, 10)
    rate_a = np.exp(log_rate_a) * sf_a[:, None]
    counts_a = sample_nb(rate_a, theta_a)

    log_rate_b = z @ W_b + intercept_b
    log_rate_b = np.clip(log_rate_b, -10, 10)
    rate_b = np.exp(log_rate_b) * sf_b[:, None]
    counts_b = sample_nb(rate_b, theta_b)

    # Write units
    unit_ids = [f"u{i:04d}" for i in range(n_units)]
    units = pd.DataFrame(
        {
            "unit_id": unit_ids,
            "group__site": unit_group,
            "env__temp": rng.normal(0, 1, size=n_units),
            "cont__day_of_year": rng.uniform(0, 365, size=n_units),
        }
    )
    units.to_parquet(units_path, index=False)

    # Features
    pd.DataFrame({"feature_id": leaves_a}).to_parquet(
        out / "assay__assay_a__features.parquet", index=False
    )
    pd.DataFrame({"feature_id": leaves_b}).to_parquet(
        out / "assay__assay_b__features.parquet", index=False
    )

    # Counts as long-form
    def long_counts(counts: np.ndarray, leaves: list[str]) -> pd.DataFrame:
        rows = []
        for i, uid in enumerate(unit_ids):
            for j, fid in enumerate(leaves):
                c = int(counts[i, j])
                if c > 0:
                    rows.append((uid, fid, c))
        return pd.DataFrame(rows, columns=["unit_id", "feature_id", "count"])

    long_counts(counts_a, leaves_a).to_parquet(
        out / "assay__assay_a__counts.parquet", index=False
    )
    long_counts(counts_b, leaves_b).to_parquet(
        out / "assay__assay_b__counts.parquet", index=False
    )

    # group variance share: var(group offsets) / (var(group) + var(z|group))
    group_var = float(np.mean([np.linalg.norm(v) ** 2 for v in group_offset.values()]) / K)
    total_var = float(z.var(axis=0).mean())
    group_share = group_var / max(total_var, 1e-9)

    truth = {
        "n_units": n_units,
        "n_groups": n_groups,
        "K": K,
        "lambda_true": lam_true,
        "sigma2_true": sigma2_true,
        "theta_true": {"assay_a": theta_a, "assay_b": theta_b},
        "group_variance_share": group_share,
        "block_coupling_indicator": True,
    }
    truth_path.write_text(json.dumps(truth, indent=2))
    return truth


@pytest.fixture(scope="session")
def synthetic_smoke_dir() -> Path:
    _generate_synthetic_smoke(SYN_DIR)
    return SYN_DIR


@pytest.fixture(scope="session")
def synthetic_smoke_truth(synthetic_smoke_dir: Path) -> dict:
    return json.loads((synthetic_smoke_dir / "truth.json").read_text())
