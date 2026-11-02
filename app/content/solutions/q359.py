from torch import nn
from torch.nn import functional as F


def q359(width, bottleneck):
    """Build a residual adapter bottleneck x+up(ReLU(down(x))). Zero-initialize the up
    projection so insertion initially preserves outputs.
    """

    class Adapter(nn.Module):

        def __init__(self):
            super().__init__()
            self.down = nn.Linear(width, bottleneck)
            self.up = nn.Linear(bottleneck, width)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

        def forward(self, x):
            return x + self.up(F.relu(self.down(x)))

    return Adapter()
