import torch
from torch import nn
from torch.nn import functional as F


def q505(input_dim, hidden=32, bottleneck=8, prototypes=16):
    """Return a small DINO projection module: two-layer MLP, unit-normalized bottleneck,
    and unit-normalized prototype rows (fixed scale one). Input [B,D] maps to [B,K].
    Both normalization operations use an epsilon for zero vectors.
    """

    class Head(nn.Module):

        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, bottleneck),
            )
            self.direction = nn.Parameter(torch.randn(prototypes, bottleneck) * 0.02)

        def forward(self, x):
            z = F.normalize(self.mlp(x), dim=-1, eps=1e-06)
            w = F.normalize(self.direction, dim=-1, eps=1e-06)
            return F.linear(z, w)

    return Head()
