import torch
from torch import nn
from torch.nn import functional as F


def q526(input_dim, output_dim=256, normalize=True):
    """Return a gated linear unit for visual embeddings: value(x)*sigmoid(gate(x)),
    optionally L2-normalized. Input [B,D] maps to [B,output_dim].
    """

    class GLU(nn.Module):

        def __init__(self):
            super().__init__()
            self.value = nn.Linear(input_dim, output_dim)
            self.gate = nn.Linear(input_dim, output_dim)

        def forward(self, x):
            z = self.value(x) * torch.sigmoid(self.gate(x))
            return F.normalize(z, dim=-1) if normalize else z

    return GLU()
