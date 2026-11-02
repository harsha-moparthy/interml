import torch
from torch import nn
from torch.nn import functional as F


def q515(base, rank=4, alpha=4):
    """Wrap a fused nn.Linear(D,3D) QKV projection with trainable rank-r updates on Q/V
    only. Freeze base weights/bias, initialize A randomly and B to zero, preserve K.
    Return the module; input may have arbitrary leading axes.
    """
    if base.out_features != 3 * base.in_features or rank < 1:
        raise ValueError("invalid QKV/rank")

    # Only Q and V receive low-rank updates; the fused K projection stays unchanged.
    class QVLoRA(nn.Module):

        def __init__(self):
            super().__init__()
            self.base = base.requires_grad_(False)
            d = base.in_features
            self.a = nn.Parameter(base.weight.new_empty(2, rank, d).normal_(std=0.02))
            self.b = nn.Parameter(base.weight.new_zeros(2, d, rank))

        def forward(self, x):
            q, k, v = self.base(x).chunk(3, dim=-1)
            dq = F.linear(F.linear(x, self.a[0]), self.b[0]) * (alpha / rank)
            dv = F.linear(F.linear(x, self.a[1]), self.b[1]) * (alpha / rank)
            return torch.cat((q + dq, k, v + dv), dim=-1)

    return QVLoRA()
