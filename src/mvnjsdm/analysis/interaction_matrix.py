"""Residual interaction matrices in latent and feature space.

Three deliverables:

* ``latent_interaction_matrix(model, dm)``: posterior covariance of the joint
  latent z across units. Captures the residual covariance left after the
  prior has been applied (the "interaction" in latent space).

* ``cross_assay_interaction_matrix(model, dm, assay_a, assay_b)``:
  ``[features_a x features_b]`` coupling between two assays' expected
  outputs. Approximated as ``J_a Sigma_z J_b^T`` where ``J_a`` is the
  decoder jacobian averaged over units and ``Sigma_z`` is the latent
  posterior covariance.

* ``within_assay_interaction_matrix(model, dm, assay)``: same construction
  but for a single assay.

The jacobian-based path captures decoder non-linearities (mean-field
linearisation at each unit's posterior mean). The ``use_jacobian=False`` path
falls back to ``W Sigma_z W^T`` using only the loading matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from ..interpret.decoder_jacobian import decoder_jacobian_dataset
from .latent_export import _unwrap_model


def _gather_posterior_z(model: Any, datamodule: Any, split: str) -> np.ndarray:
    """Return [N, K_joint] matrix of posterior mean z across the chosen split."""
    m = _unwrap_model(model)
    m.eval()
    zs: list[np.ndarray] = []
    if split == "all":
        loaders = [datamodule.train_dataloader(), datamodule.val_dataloader()]
    elif split == "train":
        loaders = [datamodule.train_dataloader()]
    elif split == "val":
        loaders = [datamodule.val_dataloader()]
    else:
        raise ValueError(f"unknown split: {split}")
    seen: set[str] = set()
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                out = m(batch)
                mu = out["mu_q"].detach().cpu().numpy()
                for i, uid in enumerate(batch["unit_id"]):
                    if uid in seen:
                        continue
                    seen.add(uid)
                    zs.append(mu[i])
    return np.stack(zs, axis=0)


def latent_interaction_matrix(model: Any, datamodule: Any, *, split: str = "all") -> np.ndarray:
    """Return ``Sigma_z``, the posterior covariance of joint z (``[K, K]``).

    Computed as the empirical covariance of the posterior-mean ``z`` across
    units. This is a residual interaction view: anything still correlated
    after prior + hierarchy reflects un-modelled coupling.
    """
    Z = _gather_posterior_z(model, datamodule, split)
    Z = Z - Z.mean(0, keepdims=True)
    n = Z.shape[0]
    cov = (Z.T @ Z) / max(n - 1, 1)
    # symmetrise (defensive against numerical asymmetry)
    cov = 0.5 * (cov + cov.T)
    return cov


def _assay_input_slice(model: Any, assay: str) -> np.ndarray:
    """Return the index array selecting [z_shared, z_private_<assay>] from joint z."""
    m = _unwrap_model(model)
    shared = m.shared_dim
    off = m.private_offsets[assay]
    d = m.private_dims[assay]
    idx = list(range(shared)) + list(range(off, off + d))
    return np.asarray(idx, dtype=np.int64)


def _decoder_W(model: Any, assay: str) -> np.ndarray:
    m = _unwrap_model(model)
    W = m.decoders[assay].loading_matrix().detach().cpu().numpy()
    return W  # [K_in, F]


def _mean_jacobian(model: Any, datamodule: Any, *, assay: str, split: str) -> np.ndarray:
    """Return the average per-unit jacobian for an assay, ``[K_in, F]``."""
    J = decoder_jacobian_dataset(model, datamodule, assay=assay, split=split)
    return J.mean(axis=0).detach().cpu().numpy()


def _feature_ids(datamodule: Any, assay: str) -> list[str]:
    ad_obj = datamodule.mud[assay]
    if "feature_id" in ad_obj.var.columns:
        return ad_obj.var["feature_id"].astype(str).tolist()
    return [str(i) for i in ad_obj.var.index]


def cross_assay_interaction_matrix(
    model: Any,
    datamodule: Any,
    *,
    assay_a: str,
    assay_b: str,
    split: str = "all",
    use_jacobian: bool = True,
) -> pd.DataFrame:
    """Return ``[features_a x features_b]`` cross-assay coupling.

    coupling = J_a^T Sigma_z J_b   (after slicing Sigma_z to the union of
    z dims that feed each decoder).

    With ``use_jacobian=False``, ``J`` is replaced by the loading matrix
    ``W`` (linear-decoder approximation).
    """
    Sigma_z = latent_interaction_matrix(model, datamodule, split=split)
    idx_a = _assay_input_slice(model, assay_a)
    idx_b = _assay_input_slice(model, assay_b)
    Sigma_ab = Sigma_z[np.ix_(idx_a, idx_b)]
    if use_jacobian:
        Ja = _mean_jacobian(model, datamodule, assay=assay_a, split=split)  # [K_in_a, F_a]
        Jb = _mean_jacobian(model, datamodule, assay=assay_b, split=split)  # [K_in_b, F_b]
    else:
        Ja = _decoder_W(model, assay_a)
        Jb = _decoder_W(model, assay_b)
    M = Ja.T @ Sigma_ab @ Jb  # [F_a, F_b]
    feats_a = _feature_ids(datamodule, assay_a)
    feats_b = _feature_ids(datamodule, assay_b)
    return pd.DataFrame(M, index=feats_a, columns=feats_b)


def within_assay_interaction_matrix(
    model: Any,
    datamodule: Any,
    *,
    assay: str,
    split: str = "all",
    use_jacobian: bool = True,
) -> pd.DataFrame:
    """Return ``[features x features]`` within-assay coupling matrix.

    coupling = J^T Sigma_z[z_in, z_in] J -- symmetric by construction.
    """
    Sigma_z = latent_interaction_matrix(model, datamodule, split=split)
    idx = _assay_input_slice(model, assay)
    Sigma_aa = Sigma_z[np.ix_(idx, idx)]
    if use_jacobian:
        J = _mean_jacobian(model, datamodule, assay=assay, split=split)  # [K_in, F]
    else:
        J = _decoder_W(model, assay)
    M = J.T @ Sigma_aa @ J
    # symmetrise (defensive)
    M = 0.5 * (M + M.T)
    feats = _feature_ids(datamodule, assay)
    return pd.DataFrame(M, index=feats, columns=feats)
