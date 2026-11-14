"""Hard gain selection with explicit baseline fallback."""

import math
import torch
from torch import nn


class Selector(nn.Module):
    def forward(self, gains, delta=0.0, fallback=True):
        if not math.isfinite(delta) or not torch.isfinite(gains).all():
            raise ValueError('Gains and threshold must be finite')
        best, index = gains.max(-1)  # Ties choose the lowest expert index.
        selected = index + 1
        if fallback:
            selected = torch.where(best > delta, selected, torch.zeros_like(selected))
        return selected
