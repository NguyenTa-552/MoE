"""Local gain targets never contain global Dice terms."""

import math
import torch
from torch.nn import functional as F


def patch_occupancy(mask, patches, classes, ignore_index=-1):
    """Area-weighted class occupancy; ignored pixels contribute no mass."""
    side = math.isqrt(patches)
    if side * side != patches:
        raise ValueError('Expected a square patch grid')
    if mask.ndim == 4:
        mask = mask.squeeze(1)
    valid = mask != ignore_index
    if ((mask[valid] < 0) | (mask[valid] >= classes)).any():
        raise ValueError('Mask IDs must index the candidate vocabulary')
    safe = mask.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe.long(), classes).permute(0, 3, 1, 2).float()
    mass = F.interpolate(one_hot * valid[:, None], (side, side), mode='area')
    coverage = F.interpolate(valid[:, None].float(), (side, side), mode='area')
    occupancy = (mass / coverage.clamp_min(1e-12)).flatten(2).transpose(1, 2)
    return occupancy, coverage.flatten(1)


def local_losses(logits, occupancy):
    """Mean layer soft-target CE: [L,B,P,K,C] -> [B,P,K]."""
    return -(logits.log_softmax(-1) * occupancy[None, ..., None, :]).sum(-1).mean(0)


def gain_targets(losses, views=None):
    gains = losses[..., :1] - losses[..., 1:]
    if views is not None:
        gains = torch.stack([gains] + [v[..., :1] - v[..., 1:] for v in views]).amin(0)
    return gains.detach()


def balanced_weights(occupancy, coverage):
    """Give normal and defective occupancy equal mass when both exist."""
    normal = occupancy[..., 0] * coverage
    defect = occupancy[..., 1:].sum(-1) * coverage
    groups = [x / x.sum() for x in (normal, defect) if x.sum() > 0]
    return sum(groups) / len(groups) if groups else coverage * 0


def gain_mse(prediction, targets, occupancy, coverage):
    weight = balanced_weights(occupancy, coverage)
    return ((prediction - targets.detach()).square().mean(-1) * weight).sum()
