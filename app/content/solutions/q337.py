from torch import nn


def q337(width, heads, hidden):
    """Construct a trainable post-norm transformer encoder layer using PyTorch module
    primitives, bidirectional self-attention, and a GELU feed-forward network. Return
    the module; input shape is (B,L,D).
    """

    class Encoder(nn.Module):

        def __init__(self):
            super().__init__()
            self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
            self.norm1 = nn.LayerNorm(width)
            self.norm2 = nn.LayerNorm(width)
            self.ff = nn.Sequential(
                nn.Linear(width, hidden),
                nn.GELU(),
                nn.Linear(hidden, width),
            )

        def forward(self, x, padding_mask=None):
            a, _ = self.attention(
                x, x, x, key_padding_mask=padding_mask, need_weights=False
            )
            x = self.norm1(x + a)
            return self.norm2(x + self.ff(x))

    return Encoder()
