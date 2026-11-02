from torch import nn


def q575(input_dim=768, width=32, classes=150):
    """Return a supervised linear bottleneck: frozen input features [B,D] ->
    width-dimensional code -> class logits. Detach input features, train both linear
    layers, and return (logits,code). The feature encoder is supplied upstream, not
    downloaded.
    """

    class Bottleneck(nn.Module):

        def __init__(self):
            super().__init__()
            self.compress = nn.Linear(input_dim, width)
            self.classify = nn.Linear(width, classes)

        def forward(self, x):
            z = self.compress(x.detach())
            return (self.classify(z), z)

    return Bottleneck()
