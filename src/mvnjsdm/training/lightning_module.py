"""LightningModule wrapping MVNeuralJSDM."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch

from ..models.mvnjsdm import MVNeuralJSDM
from .losses import LossWeights, compute_loss


class MVNeuralJSDMLit(pl.LightningModule):
    def __init__(
        self,
        model: MVNeuralJSDM,
        weights: LossWeights | None = None,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.model = model
        self.weights = weights or LossWeights()
        self.lr = lr
        # avoid storing module ref via hyperparameters
        self.save_hyperparameters(ignore=["model", "weights"])

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.model(batch)

    def _step(self, batch: dict[str, Any], stage: str) -> dict[str, Any]:
        out = self.model(batch)
        parts = compute_loss(self.model, batch, out, self.weights)
        for k, v in parts.items():
            self.log(f"{stage}/{k}", v.detach(), prog_bar=(k == "loss"), batch_size=out["mu_q"].shape[0])
        return {"loss": parts["loss"], "mu_q": out["mu_q"]}

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")["loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
