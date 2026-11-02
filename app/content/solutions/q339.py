import torch
from torch import nn
from torch.nn import functional as F


def q339(vocabulary, width=64, heads=4, layers=2, context=128):
    """Build a tiny GPT with learned token/position embeddings, causal pre-norm transformer
    layers, final normalization, and tied output weights. Return logits (B,L,V); no
    pretrained weights or downloads are used.
    """

    class GPT(nn.Module):

        def __init__(self):
            super().__init__()
            self.tokens = nn.Embedding(vocabulary, width)
            self.positions = nn.Embedding(context, width)
            self.blocks = nn.ModuleList(
                (
                    nn.TransformerEncoderLayer(
                        width,
                        heads,
                        4 * width,
                        dropout=0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(layers)
                )
            )
            self.norm = nn.LayerNorm(width)

        def forward(self, ids):
            length = ids.shape[1]
            if length > context:
                raise ValueError("context exceeded")
            x = (
                self.tokens(ids)
                + self.positions(torch.arange(length, device=ids.device))[None]
            )
            mask = torch.triu(
                torch.ones(length, length, device=ids.device, dtype=torch.bool),
                1,
            )
            for block in self.blocks:
                x = block(x, src_mask=mask)
            return F.linear(self.norm(x), self.tokens.weight)

    return GPT()
