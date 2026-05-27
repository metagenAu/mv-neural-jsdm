"""Decoder Jacobians ``dE[x]/dz`` for interpretation and cross-assay coupling.

For a decoder whose forward returns ``LikelihoodParams`` with ``mu`` of shape
``[B, F]`` and input ``z`` of shape ``[B, K_in]``, the per-unit Jacobian is
``[K_in, F]``. We use ``torch.autograd.functional.jacobian`` to stay
decoder-agnostic (works for NB / Gaussian / Bernoulli alike).

For Bernoulli decoders we report ``d sigmoid(logits)/dz`` so the result is in
probability space; for NB we get ``d rate/dz``; for Gaussian, ``dmu/dz``.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.autograd.functional import jacobian


def _unwrap_model(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "decoders"):
        return model.model
    return model


def _expected_response_fn(model: Any, assay: str, size_factor: Tensor | None):
    """Return a callable z_in -> E[x] for a single z_in (1-D tensor)."""
    m = _unwrap_model(model)
    decoder = m.decoders[assay]
    likelihood = m.likelihoods[assay].__class__.__name__

    def fn(z_in: Tensor) -> Tensor:
        z2 = z_in.unsqueeze(0)
        sf = size_factor.unsqueeze(0) if size_factor is not None else None
        params = decoder(z2, size_factor=sf)
        mu = params.mu
        if likelihood == "BernoulliLikelihood":
            mu = torch.sigmoid(mu)
        return mu.squeeze(0)

    return fn


def decoder_jacobian(
    model: Any,
    *,
    assay: str,
    z_ref: Tensor | None = None,
    batch_cov: Tensor | None = None,
    size_factor: Tensor | None = None,
) -> Tensor:
    """Return ``[K_in, F]`` Jacobian of ``E[x_assay]`` wrt the decoder input z.

    Parameters
    ----------
    z_ref
        1-D ``[K_in]`` tensor. If ``None``, zeros of the right shape are used.
    size_factor
        Scalar tensor for NB-style decoders; ignored otherwise.
    """
    m = _unwrap_model(model)
    decoder = m.decoders[assay]
    K_in = decoder.in_dim
    if z_ref is None:
        z_ref = torch.zeros(K_in, dtype=torch.float32)
    fn = _expected_response_fn(m, assay, size_factor)
    # jacobian returns [F, K_in]; transpose to [K_in, F]
    J = jacobian(fn, z_ref.detach(), vectorize=True)
    return J.transpose(0, 1).contiguous()


def _gather_z_and_sf(
    model: Any, datamodule: Any, *, assay: str, split: str
) -> tuple[Tensor, list[Tensor | None]]:
    m = _unwrap_model(model)
    shared = m.shared_dim
    off = m.private_offsets[assay]
    d = m.private_dims[assay]
    if split == "all":
        loaders = [datamodule.train_dataloader(), datamodule.val_dataloader()]
    elif split == "train":
        loaders = [datamodule.train_dataloader()]
    elif split == "val":
        loaders = [datamodule.val_dataloader()]
    else:
        raise ValueError(f"unknown split: {split}")
    z_rows: list[Tensor] = []
    sfs: list[Tensor | None] = []
    seen: set[str] = set()
    m.eval()
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                out = m(batch)
                mu = out["mu_q"].detach()
                sf_b = batch["assays"][assay].get("sf", None)
                for i, uid in enumerate(batch["unit_id"]):
                    if uid in seen:
                        continue
                    seen.add(uid)
                    z_full = mu[i]
                    z_in = torch.cat([z_full[:shared], z_full[off : off + d]], dim=0)
                    z_rows.append(z_in)
                    sfs.append(sf_b[i].detach() if sf_b is not None else None)
    Z = torch.stack(z_rows, dim=0)
    return Z, sfs


def decoder_jacobian_dataset(
    model: Any,
    datamodule: Any,
    *,
    assay: str,
    split: str = "all",
) -> Tensor:
    """Per-unit decoder Jacobians for an assay, ``[N, K_in, F]``.

    Useful as the input to callers that average or weight by unit.
    """
    Z, sfs = _gather_z_and_sf(model, datamodule, assay=assay, split=split)
    Js: list[Tensor] = []
    for i in range(Z.shape[0]):
        J = decoder_jacobian(model, assay=assay, z_ref=Z[i], size_factor=sfs[i])
        Js.append(J)
    return torch.stack(Js, dim=0)
