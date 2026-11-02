import numpy as np
from torch import nn
from torch.nn import functional as F


def q358(module, rank=4, alpha=8):
    """Recursively replace each nn.Linear child with a frozen base plus trainable LoRA
    adapter. Initialize B to zero so the initial model is unchanged. Preserve
    device/dtype.
    """

    class LoRALinear(nn.Module):

        def __init__(self, base):
            super().__init__()
            self.base = base
            for p in base.parameters():
                p.requires_grad_(False)
            self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
            nn.init.kaiming_uniform_(self.a, a=np.sqrt(5))
            self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))

        def forward(self, x):
            return self.base(x) + alpha / rank * F.linear(F.linear(x, self.a), self.b)

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child))
        else:
            q358(child, rank, alpha)
    return module
