"""Feature-structure adapters: Newick trees and edge-list graphs.

We expose minimal NumPy arrays:

    tree_variance_covariance(newick) -> (feature_ids, C)  where
        C[i,j] = shared branch length from root down to mrca(i, j)

    graph_laplacian(edges_df, feature_ids) -> L  combinatorial Laplacian L = D - A

We try ete3 first and fall back to a manual parser if ete3 import fails or
parsing errors.  The fallback supports a sensible subset of Newick: nested
parentheses, branch lengths after ':', leaf labels, internal labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Newick parsing
# ---------------------------------------------------------------------------


class _Node:
    __slots__ = ("name", "length", "children", "parent")

    def __init__(self, name: str = "", length: float = 0.0):
        self.name = name
        self.length = length
        self.children: list[_Node] = []
        self.parent: _Node | None = None


def _parse_newick_manual(s: str) -> _Node:
    s = s.strip()
    if s.endswith(";"):
        s = s[:-1]

    i = 0
    n = len(s)

    def parse_subtree() -> _Node:
        nonlocal i
        node = _Node()
        if i < n and s[i] == "(":
            i += 1
            while True:
                child = parse_subtree()
                child.parent = node
                node.children.append(child)
                if i < n and s[i] == ",":
                    i += 1
                    continue
                if i < n and s[i] == ")":
                    i += 1
                    break
                raise ValueError(f"unexpected char at {i}: {s[i:i+10]!r}")
        # name
        start = i
        while i < n and s[i] not in "(),:":
            i += 1
        node.name = s[start:i]
        # length
        if i < n and s[i] == ":":
            i += 1
            start = i
            while i < n and s[i] not in "(),":
                i += 1
            try:
                node.length = float(s[start:i])
            except ValueError:
                node.length = 0.0
        return node

    root = parse_subtree()
    return root


def _parse_newick(newick: str) -> _Node:
    try:
        from ete3 import Tree  # type: ignore

        t = Tree(newick, format=1)

        def convert(et_node) -> _Node:
            n = _Node(name=et_node.name or "", length=float(et_node.dist or 0.0))
            for c in et_node.children:
                child = convert(c)
                child.parent = n
                n.children.append(child)
            return n

        return convert(t)
    except Exception:
        return _parse_newick_manual(newick)


def _leaves(root: _Node) -> list[_Node]:
    out: list[_Node] = []
    stack = [root]
    while stack:
        x = stack.pop()
        if not x.children:
            out.append(x)
        else:
            stack.extend(x.children)
    out.reverse()
    return out


def _root_distance(node: _Node) -> float:
    d = 0.0
    cur: _Node | None = node
    while cur is not None and cur.parent is not None:
        d += cur.length
        cur = cur.parent
    return d


def _ancestors(node: _Node) -> list[_Node]:
    out = []
    cur: _Node | None = node
    while cur is not None:
        out.append(cur)
        cur = cur.parent
    return out


def tree_variance_covariance(
    newick_or_path: str | Path,
) -> tuple[list[str], np.ndarray]:
    """Compute the (Brownian-motion) tree variance-covariance matrix.

    C[i,j] = sum of branch lengths from root to most-recent common ancestor of
    leaves i and j (i.e. shared evolutionary path length). Diagonal C[i,i] is
    the root-to-leaf distance.

    Returns the leaf names (in matrix order) and the matrix.
    """
    if isinstance(newick_or_path, Path):
        newick = newick_or_path.read_text()
    elif isinstance(newick_or_path, str) and "(" not in newick_or_path and len(newick_or_path) < 1024:
        # treat as path only if it looks like a path
        p = Path(newick_or_path)
        if p.exists():
            newick = p.read_text()
        else:
            newick = newick_or_path
    else:
        newick = str(newick_or_path)

    root = _parse_newick(newick)
    leaves = _leaves(root)
    names = [leaf.name for leaf in leaves]
    F = len(leaves)
    C = np.zeros((F, F), dtype=np.float64)

    # depth from root of each ancestor node (computed once)
    # We compute MRCA by following parent pointers; ancestors lists are short.
    leaf_ancestors = [_ancestors(leaf) for leaf in leaves]
    leaf_ancestor_sets = [set(id(a) for a in anc) for anc in leaf_ancestors]
    # cache: distance from root to each node
    distances: dict[int, float] = {}

    def dist_root(node: _Node) -> float:
        k = id(node)
        if k in distances:
            return distances[k]
        d = _root_distance(node)
        distances[k] = d
        return d

    for i in range(F):
        for j in range(i, F):
            # MRCA: first ancestor of leaves[i] that is also an ancestor of leaves[j]
            mrca = None
            for a in leaf_ancestors[i]:
                if id(a) in leaf_ancestor_sets[j]:
                    mrca = a
                    break
            v = dist_root(mrca) if mrca is not None else 0.0
            C[i, j] = v
            C[j, i] = v

    return names, C


# ---------------------------------------------------------------------------
# Graph Laplacian
# ---------------------------------------------------------------------------


def graph_laplacian(
    edges_df: pd.DataFrame,
    feature_ids: list[str],
    weight_col: str | None = None,
) -> np.ndarray:
    """Combinatorial Laplacian L = D - A for the (sparse) graph given by edges.

    ``edges_df`` is a long table with columns ``source`` and ``target`` and an
    optional weight column. ``feature_ids`` defines the ordering of rows/cols.
    """
    idx = {f: i for i, f in enumerate(feature_ids)}
    F = len(feature_ids)
    A = np.zeros((F, F), dtype=np.float64)
    for _, row in edges_df.iterrows():
        i = idx.get(row["source"])
        j = idx.get(row["target"])
        if i is None or j is None or i == j:
            continue
        w = float(row[weight_col]) if weight_col else 1.0
        A[i, j] += w
        A[j, i] += w
    D = np.diag(A.sum(axis=1))
    return D - A
