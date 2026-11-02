import torch
from torch import nn


def q046(width):
    """Build a module with a trainable scale vector and a nontrainable offset buffer. Its
    output is x*scale+offset; initialize scale to one and offset to zero.
    """

    class Affine(nn.Module):

        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(width))
            self.register_buffer("offset", torch.zeros(width))

        def forward(self, x):
            return x * self.scale + self.offset

    return Affine()
