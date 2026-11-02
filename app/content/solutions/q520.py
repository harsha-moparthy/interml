import torch
from torch import nn
from torch.nn import functional as F


def q520(channels, growth=8, depth=3, scale=0.2):
    """Return a dense residual group for BCHW inputs: each 3x3 block sees concatenated
    earlier features, followed by a 1x1 channel fusion and scaled residual. This
    isolates DRCT-style feature connectivity; it does not include attention.
    """

    class Group(nn.Module):

        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList(
                (
                    nn.Conv2d(channels + i * growth, growth, 3, padding=1)
                    for i in range(depth)
                )
            )
            self.fuse = nn.Conv2d(channels + depth * growth, channels, 1)

        def forward(self, x):
            features = [x]
            for layer in self.blocks:
                features.append(F.gelu(layer(torch.cat(features, 1))))
            return x + scale * self.fuse(torch.cat(features, 1))

    return Group()
