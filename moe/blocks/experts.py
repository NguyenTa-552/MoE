"""Residual adapters; expert zero is supplied by the frozen baseline."""

import torch
from torch import nn


class ResidualExpert(nn.Module):
    def __init__(self, input_dim, output_dim, rank=16, alpha=0.1):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)
        self.activation = nn.GELU()
        self.alpha = alpha
        nn.init.zeros_(self.up.weight)

    def forward(self, h, z0):
        return z0.detach() + self.alpha * self.up(self.activation(self.down(h.detach())))


class ExpertBank(nn.Module):
    def __init__(self, input_dim, output_dim, layers=4, experts=3, rank=16, alpha=0.1):
        super().__init__()
        if min(input_dim, output_dim, layers, experts, rank) < 1:
            raise ValueError('Expert dimensions must be positive')
        self.experts = nn.ModuleList([
            nn.ModuleList([ResidualExpert(input_dim, output_dim, rank, alpha)
                           for _ in range(layers)]) for _ in range(experts)
        ])

    def forward(self, h, z0):
        # [L,B,P,Dv], [L,B,P,D] -> [L,B,P,E,D]
        if len(h) != len(self.experts[0]) or len(z0) != len(h):
            raise ValueError('Feature layers do not match expert bank')
        return torch.stack([
            torch.stack([adapter(h[i], z0[i]) for i, adapter in enumerate(expert)])
            for expert in self.experts
        ], dim=-2)

    def forward_expert(self, index, h, z0):
        return torch.stack([adapter(h[i], z0[i])
                            for i, adapter in enumerate(self.experts[index])])
