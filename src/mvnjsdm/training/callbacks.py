"""Training callbacks."""

from __future__ import annotations

import pytorch_lightning as pl
import torch


class LatentCollapseMonitor(pl.Callback):
    """Logs the per-latent-dim variance of mu_q each epoch.

    A dim with very small variance across the dataset is effectively collapsed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mus: list[torch.Tensor] = []

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ) -> None:
        if outputs is None:
            return
        mu = outputs.get("mu_q")
        if mu is not None:
            self._mus.append(mu.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._mus:
            return
        mu = torch.cat(self._mus, dim=0)
        variances = mu.var(dim=0)
        active = (variances > 1e-2).sum().item()
        pl_module.log("latent/active_dims", float(active), prog_bar=False)
        pl_module.log("latent/mean_var", variances.mean().item(), prog_bar=False)
        self._mus.clear()
