import torch
from torch import nn


def q287(image_size, patch, channels, width, heads, layers):
    """Build a small ViT image encoder with convolutional patch projection, learned
    class/position tokens, and PyTorch TransformerEncoder layers. Return the normalized
    class-token embedding.
    """

    class ViT(nn.Module):

        def __init__(self):
            super().__init__()
            count = (image_size // patch) ** 2
            self.patch = nn.Conv2d(channels, width, patch, stride=patch)
            self.cls = nn.Parameter(torch.zeros(1, 1, width))
            self.pos = nn.Parameter(torch.zeros(1, count + 1, width))
            layer = nn.TransformerEncoderLayer(
                width,
                heads,
                4 * width,
                dropout=0,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, layers)
            self.norm = nn.LayerNorm(width)

        def forward(self, x):
            tokens = self.patch(x).flatten(2).transpose(1, 2)
            tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], 1) + self.pos
            return self.norm(self.encoder(tokens)[:, 0])

    return ViT()
