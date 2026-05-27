"""Neural AR transition over an ordered continuous index. Skeleton only.

``ar.index_name`` in the config references a ``cont__<name>`` column.
"""

from __future__ import annotations

import torch.nn as nn


class NeuralARTransition(nn.Module):  # skeleton
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover - skeleton
        raise NotImplementedError("NeuralARTransition is a skeleton")
