from __future__ import annotations

import numpy as np
import pandas as pd

from mvnjsdm.data.feature_structures import graph_laplacian, tree_variance_covariance


def test_tree_covariance_simple():
    # ((A:1,B:1):1,C:2);
    newick = "((A:1,B:1):1,C:2);"
    names, C = tree_variance_covariance(newick)
    assert sorted(names) == ["A", "B", "C"]
    # Build a name->idx map
    idx = {n: i for i, n in enumerate(names)}
    # A-B share an internal branch of length 1 above their mrca
    assert np.isclose(C[idx["A"], idx["B"]], 1.0)
    # A-A diagonal = 2 (1 internal + 1 leaf)
    assert np.isclose(C[idx["A"], idx["A"]], 2.0)
    # A-C share only the root -> 0
    assert np.isclose(C[idx["A"], idx["C"]], 0.0)
    # symmetric
    assert np.allclose(C, C.T)


def test_graph_laplacian_simple():
    edges = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"]})
    L = graph_laplacian(edges, ["a", "b", "c"])
    # Each node degree + L = D - A
    assert L.shape == (3, 3)
    assert np.isclose(L.sum(), 0.0)
    assert np.allclose(L, L.T)
    # diag = degrees
    assert L[0, 0] == 1
    assert L[1, 1] == 2
    assert L[2, 2] == 1
