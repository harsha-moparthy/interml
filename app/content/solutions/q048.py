from torch import nn
from torch.nn import functional as F


def q048(width, depth):
    """Build a stack of depth registered linear layers using ModuleList. Apply ReLU after
    each layer so all parameters appear in state_dict.
    """

    class Stack(nn.Module):

        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList((nn.Linear(width, width) for _ in range(depth)))

        def forward(self, x):
            for layer in self.layers:
                x = F.relu(layer(x))
            return x

    return Stack()
