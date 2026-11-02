from torch import nn


def q043(features):
    """Build an nn.Module for linear regression from features inputs to one output,
    including a learned bias.
    """

    class Regression(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(features, 1)

        def forward(self, x):
            return self.linear(x)

    return Regression()
