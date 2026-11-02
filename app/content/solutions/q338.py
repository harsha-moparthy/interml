import torch
from torch import nn


def q338(width, heads, hidden):
    """Construct an encoder-decoder transformer decoder layer with causal self-attention,
    cross-attention to encoder memory, and a feed-forward network. Use post-norm and no
    dropout. True in memory_padding means padding.
    """

    class Decoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.norms = nn.ModuleList((nn.LayerNorm(width) for _ in range(3)))
            self.ff = nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Linear(hidden, width),
            )

        def forward(self, x, memory, memory_padding=None):
            blocked = torch.triu(
                torch.ones(
                    x.shape[1],
                    x.shape[1],
                    device=x.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            a, _ = self.self_attention(x, x, x, attn_mask=blocked, need_weights=False)
            x = self.norms[0](x + a)
            a, _ = self.cross_attention(
                x,
                memory,
                memory,
                key_padding_mask=memory_padding,
                need_weights=False,
            )
            x = self.norms[1](x + a)
            return self.norms[2](x + self.ff(x))

    return Decoder()
