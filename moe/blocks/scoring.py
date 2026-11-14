"""DAPO-compatible normalization and score-space operations."""

import math
from torch import nn
from torch.nn import functional as F


class BaselineScorer(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError('Temperature must be positive')
        self.temperature = temperature

    def forward(self, features, text):
        return F.normalize(features, dim=-1) @ F.normalize(text, dim=-1).T / self.temperature


def dense_maps(probabilities, size):
    """[L,B,P,C] probabilities -> [L,B,C,H,W], softmax already applied."""
    layers, batch, patches, classes = probabilities.shape
    side = math.isqrt(patches)
    if side * side != patches:
        raise ValueError('DAPO requires a square patch grid')
    grid = probabilities.permute(0, 1, 3, 2).reshape(layers * batch, classes, side, side)
    return F.interpolate(grid, size=size, mode='bilinear', align_corners=True).reshape(
        layers, batch, classes, *size)


def select_probabilities(probabilities, indices):
    """[L,B,P,K,C], [B,P] -> [L,B,P,C]; K includes baseline."""
    index = indices[None, ..., None, None].expand(
        probabilities.shape[0], *indices.shape, 1, probabilities.shape[-1])
    return probabilities.gather(-2, index).squeeze(-2)
