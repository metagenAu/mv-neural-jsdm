"""Cross-assay coupling — contemporaneous and lagged.

* :func:`contemporaneous_coupling` samples z from q(z|x) repeatedly per unit
  and correlates decoded outputs across assays at the same index point. Works
  with any trained model.

* :func:`lagged_coupling` correlates decoded(z(t)) with decoded(z(t+lag))
  using the GP posterior to propagate z smoothly along the continuous index.
  Requires a model with a GP prior (raises a clear error otherwise).

* :func:`latent_lagged_covariance` returns the K_joint x K_joint cross-cov of
  z at t vs t+lag from the GP posterior.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from ..interpret.decoder_jacobian import _unwrap_model


def _gather_posterior(model: Any, datamodule: Any) -> dict[str, Any]:
    """Return per-unit posterior mean/logvar plus cont indices and size factors."""
    m = _unwrap_model(model)
    m.eval()
    rows: dict[str, dict[str, Any]] = {}
    loaders = [datamodule.train_dataloader(), datamodule.val_dataloader()]
    with torch.no_grad():
        for loader in loaders:
            for batch in loader:
                out = m(batch)
                mu = out["mu_q"].detach()
                lv = out["logvar_q"].detach()
                cont = batch.get("cont")
                cont_cols = batch.get("cont_columns") or []
                for i, uid in enumerate(batch["unit_id"]):
                    if uid in rows:
                        continue
                    rec = {"mu": mu[i], "logvar": lv[i]}
                    if cont is not None:
                        rec["cont"] = cont[i]
                        rec["cont_columns"] = list(cont_cols)
                    for asy, vals in batch["assays"].items():
                        if vals.get("sf") is not None:
                            rec.setdefault("sf", {})[asy] = vals["sf"][i].detach()
                    rows[uid] = rec
    return rows


def _decoded_mean(model: Any, assay: str, z_in: torch.Tensor, size_factor: torch.Tensor | None) -> torch.Tensor:
    """Return decoded expected output [N, F] for the given assay."""
    m = _unwrap_model(model)
    decoder = m.decoders[assay]
    likelihood = m.likelihoods[assay].__class__.__name__
    with torch.no_grad():
        params = decoder(z_in, size_factor=size_factor)
    mu = params.mu
    if likelihood == "BernoulliLikelihood":
        mu = torch.sigmoid(mu)
    return mu


def _assay_z_in(m: Any, assay: str, z_joint: torch.Tensor) -> torch.Tensor:
    """Slice [B, joint] -> [B, shared+private_a] for the given assay's decoder."""
    shared = m.shared_dim
    off = m.private_offsets[assay]
    d = m.private_dims[assay]
    return torch.cat([z_joint[:, :shared], z_joint[:, off:off + d]], dim=-1)


def contemporaneous_coupling(
    model: Any,
    datamodule: Any,
    *,
    assay_a: str,
    assay_b: str,
    n_samples: int = 32,
) -> pd.DataFrame:
    """Cross-assay coupling at the same index: corr(dec_a(z), dec_b(z)).

    For each unit we sample n_samples z's from q(z|x), decode both assays, then
    pool all (sample, unit) decoded outputs and compute Pearson correlations
    column-wise. Returns ``[F_a, F_b]`` correlation DataFrame.
    """
    m = _unwrap_model(model)
    posterior = _gather_posterior(model, datamodule)
    if len(posterior) == 0:
        raise RuntimeError("no posterior samples collected from datamodule")

    mu_all = torch.stack([r["mu"] for r in posterior.values()], dim=0)  # [N, K]
    lv_all = torch.stack([r["logvar"] for r in posterior.values()], dim=0)
    sfs_a = None
    sfs_b = None
    if any("sf" in r and assay_a in r.get("sf", {}) for r in posterior.values()):
        sfs_a = torch.stack(
            [r.get("sf", {}).get(assay_a, torch.tensor(1.0)) for r in posterior.values()],
            dim=0,
        )
    if any("sf" in r and assay_b in r.get("sf", {}) for r in posterior.values()):
        sfs_b = torch.stack(
            [r.get("sf", {}).get(assay_b, torch.tensor(1.0)) for r in posterior.values()],
            dim=0,
        )

    decoded_a_list = []
    decoded_b_list = []
    for _ in range(n_samples):
        eps = torch.randn_like(mu_all)
        z = mu_all + (0.5 * lv_all).exp() * eps
        z_a = _assay_z_in(m, assay_a, z)
        z_b = _assay_z_in(m, assay_b, z)
        decoded_a_list.append(_decoded_mean(model, assay_a, z_a, sfs_a))
        decoded_b_list.append(_decoded_mean(model, assay_b, z_b, sfs_b))
    A = torch.cat(decoded_a_list, dim=0).numpy()  # [n_samples*N, F_a]
    B = torch.cat(decoded_b_list, dim=0).numpy()

    corr = _pearson_cross(A, B)
    feat_a = list(datamodule.mud[assay_a].var_names)
    feat_b = list(datamodule.mud[assay_b].var_names)
    return pd.DataFrame(corr, index=feat_a, columns=feat_b)


def _pearson_cross(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between every column of A and every column of B."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    sa = A.std(axis=0, keepdims=True) + 1e-12
    sb = B.std(axis=0, keepdims=True) + 1e-12
    An = A / sa
    Bn = B / sb
    n = A.shape[0]
    return (An.T @ Bn) / max(n - 1, 1)


def _require_gp(model: Any) -> None:
    m = _unwrap_model(model)
    if not getattr(m, "has_gp", False):
        raise RuntimeError(
            "lagged coupling requires a GP-trained model; configure "
            "model.gp (e.g. gp=phenological) and retrain. See "
            "configs/gp/phenological.yaml for an example."
        )


def _gp_indices_for_units(model: Any, posterior: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (mu [N,K], indices [N, d], unit_ids) for units with the GP index columns."""
    m = _unwrap_model(model)
    cols = m.gp_spec.input_columns or []
    units: list[str] = []
    mus: list[torch.Tensor] = []
    idx: list[list[float]] = []
    for uid, rec in posterior.items():
        if "cont" not in rec:
            continue
        names = rec.get("cont_columns", [])
        try:
            row = [float(rec["cont"][names.index(c)]) for c in cols]
        except (ValueError, IndexError):
            continue
        units.append(uid)
        mus.append(rec["mu"])
        idx.append(row)
    if not units:
        raise RuntimeError("no posterior samples with the configured GP index columns")
    mu_arr = torch.stack(mus, dim=0).numpy()
    idx_arr = np.asarray(idx, dtype=np.float64)
    return mu_arr, idx_arr, units


def _gp_posterior_predict(
    gp_prior,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    pred_x: torch.Tensor,
    dim_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Posterior mean + variance at pred_x for one GP dim.

    Computed in zero-mean form (the prior mean is 0): K_*x [K_xx]^-1 y.
    """
    ls = gp_prior.lengthscale(dim_index)
    os_ = gp_prior.outputscale(dim_index)
    if gp_prior.kernel_name == "periodic":
        p = gp_prior.period(dim_index)
        K_xx = gp_prior._kernel_fn(train_x, train_x, ls, os_, p)
        K_sx = gp_prior._kernel_fn(pred_x, train_x, ls, os_, p)
        K_ss_diag = gp_prior._kernel_fn(pred_x, pred_x, ls, os_, p).diagonal()
    else:
        K_xx = gp_prior._kernel_fn(train_x, train_x, ls, os_)
        K_sx = gp_prior._kernel_fn(pred_x, train_x, ls, os_)
        K_ss_diag = gp_prior._kernel_fn(pred_x, pred_x, ls, os_).diagonal()
    K_xx = K_xx + gp_prior.jitter * torch.eye(K_xx.shape[0], dtype=K_xx.dtype, device=K_xx.device)
    L = torch.linalg.cholesky(K_xx)
    alpha = torch.cholesky_solve(train_y.unsqueeze(-1), L).squeeze(-1)
    mu_star = K_sx @ alpha
    v = torch.linalg.solve_triangular(L, K_sx.transpose(-1, -2), upper=False)
    var_star = (K_ss_diag - (v * v).sum(dim=0)).clamp_min(0.0)
    return mu_star, var_star


def latent_lagged_covariance(
    model: Any,
    datamodule: Any,
    *,
    index_name: str,
    lag: float,
) -> np.ndarray:
    """Return [K_joint, K_joint] cross-covariance of z(t) and z(t+lag).

    Approximation: per indexed dim, use the GP posterior mean to compute z(t)
    and z(t+lag) over the unit cloud; for non-GP dims, use the empirical
    posterior mean variance (lag has no smooth effect for these dims so we
    treat them as cov(z, z) shrunk by 0 at non-zero lag).
    """
    _require_gp(model)
    m = _unwrap_model(model)
    posterior = _gather_posterior(model, datamodule)
    mu_arr, idx_arr, _ = _gp_indices_for_units(m, posterior)
    cols = m.gp_spec.input_columns or []
    if index_name not in cols:
        raise ValueError(
            f"index_name {index_name!r} not in GP input_columns {cols!r}"
        )
    axis = cols.index(index_name)
    K_joint = mu_arr.shape[1]
    a, b = m.gp_spec.dim_start, m.gp_spec.dim_start + m.gp_spec.n_dims

    # Predict z(t) and z(t+lag) for the GP dims using each unit's index as t.
    train_x = torch.tensor(idx_arr, dtype=torch.float32)
    pred_x = train_x.clone()
    pred_x[:, axis] = pred_x[:, axis] + float(lag)

    gp_prior = None
    for sub, sl in zip(m.prior.priors, m.prior.dim_slices):
        if sl == (a, b):
            gp_prior = sub
            break
    if gp_prior is None:  # pragma: no cover - defensive
        raise RuntimeError("could not locate GP prior in the composite")

    N = mu_arr.shape[0]
    z_t = np.zeros((N, K_joint), dtype=np.float64)
    z_lag = np.zeros((N, K_joint), dtype=np.float64)
    # All dims: start from posterior mean for z(t)
    z_t[:] = mu_arr
    z_lag[:] = mu_arr  # default for non-GP dims (no smoothing)
    for k in range(b - a):
        train_y = torch.tensor(mu_arr[:, a + k], dtype=torch.float32)
        # z(t) at the unit's own index: posterior smoothing barely changes mu
        mu_t, _ = _gp_posterior_predict(gp_prior, train_x, train_y, train_x, k)
        mu_s, _ = _gp_posterior_predict(gp_prior, train_x, train_y, pred_x, k)
        z_t[:, a + k] = mu_t.detach().cpu().numpy()
        z_lag[:, a + k] = mu_s.detach().cpu().numpy()

    z_t_c = z_t - z_t.mean(axis=0, keepdims=True)
    z_lag_c = z_lag - z_lag.mean(axis=0, keepdims=True)
    return (z_t_c.T @ z_lag_c) / max(N - 1, 1)


def lagged_coupling(
    model: Any,
    datamodule: Any,
    *,
    assay_a: str,
    assay_b: str,
    index_name: str,
    lag: float,
    n_samples: int = 32,
) -> pd.DataFrame:
    """Cross-assay coupling at lag: corr(dec_a(z(t)), dec_b(z(t+lag))).

    Requires a GP-trained model. For lag=0 this delegates to
    :func:`contemporaneous_coupling`.
    """
    if abs(float(lag)) < 1e-12:
        return contemporaneous_coupling(
            model, datamodule, assay_a=assay_a, assay_b=assay_b, n_samples=n_samples
        )
    _require_gp(model)
    m = _unwrap_model(model)
    posterior = _gather_posterior(model, datamodule)
    mu_arr, idx_arr, units = _gp_indices_for_units(m, posterior)
    cols = m.gp_spec.input_columns or []
    if index_name not in cols:
        raise ValueError(
            f"index_name {index_name!r} not in GP input_columns {cols!r}"
        )
    axis = cols.index(index_name)
    a, b = m.gp_spec.dim_start, m.gp_spec.dim_start + m.gp_spec.n_dims

    train_x = torch.tensor(idx_arr, dtype=torch.float32)
    pred_x = train_x.clone()
    pred_x[:, axis] = pred_x[:, axis] + float(lag)

    # Locate GP prior
    gp_prior = None
    for sub, sl in zip(m.prior.priors, m.prior.dim_slices):
        if sl == (a, b):
            gp_prior = sub
            break
    if gp_prior is None:  # pragma: no cover - defensive
        raise RuntimeError("could not locate GP prior in the composite")

    # Predict GP-dim means + variances at t and t+lag.
    K_joint = mu_arr.shape[1]
    mu_t = torch.tensor(mu_arr, dtype=torch.float32).clone()
    var_t = torch.zeros_like(mu_t)
    mu_lag = mu_t.clone()
    var_lag = var_t.clone()
    for k in range(b - a):
        ty = torch.tensor(mu_arr[:, a + k], dtype=torch.float32)
        m_t, v_t = _gp_posterior_predict(gp_prior, train_x, ty, train_x, k)
        m_s, v_s = _gp_posterior_predict(gp_prior, train_x, ty, pred_x, k)
        mu_t[:, a + k] = m_t
        var_t[:, a + k] = v_t
        mu_lag[:, a + k] = m_s
        var_lag[:, a + k] = v_s

    # Posterior q_var on non-GP dims still varies — use the per-unit logvar.
    lv_all = torch.stack([posterior[uid]["logvar"] for uid in units], dim=0)
    q_var = lv_all.exp()
    # Mask GP dims (replace with GP-posterior variance) for sampling.
    var_t_full = q_var.clone()
    var_lag_full = q_var.clone()
    var_t_full[:, a:b] = var_t[:, a:b]
    var_lag_full[:, a:b] = var_lag[:, a:b]

    sfs_a = None
    sfs_b = None
    sf_any = any("sf" in posterior[uid] for uid in units)
    if sf_any:
        sfs_a = torch.stack(
            [posterior[uid].get("sf", {}).get(assay_a, torch.tensor(1.0)) for uid in units],
            dim=0,
        )
        sfs_b = torch.stack(
            [posterior[uid].get("sf", {}).get(assay_b, torch.tensor(1.0)) for uid in units],
            dim=0,
        )

    decoded_a_list = []
    decoded_b_list = []
    for _ in range(n_samples):
        eps_t = torch.randn_like(mu_t)
        eps_s = torch.randn_like(mu_lag)
        z_t = mu_t + var_t_full.sqrt() * eps_t
        z_s = mu_lag + var_lag_full.sqrt() * eps_s
        z_a = _assay_z_in(m, assay_a, z_t)
        z_b = _assay_z_in(m, assay_b, z_s)
        decoded_a_list.append(_decoded_mean(model, assay_a, z_a, sfs_a))
        decoded_b_list.append(_decoded_mean(model, assay_b, z_b, sfs_b))
    A = torch.cat(decoded_a_list, dim=0).numpy()
    B = torch.cat(decoded_b_list, dim=0).numpy()

    corr = _pearson_cross(A, B)
    feat_a = list(datamodule.mud[assay_a].var_names)
    feat_b = list(datamodule.mud[assay_b].var_names)
    return pd.DataFrame(corr, index=feat_a, columns=feat_b)
