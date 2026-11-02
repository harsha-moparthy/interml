import torch
from torch.nn import functional as F


def q272(x, reduce_weight, expand_weight):
    """Implement squeeze-and-excitation: global spatial pooling, a ReLU bottleneck, sigmoid
    channel gates, and rescaling of NCHW activations. Weights use input-by-output
    layout.
    """
    summary = x.mean(dim=(-2, -1))
    gates = torch.sigmoid(F.relu(summary @ reduce_weight) @ expand_weight)
    return x * gates[:, :, None, None]
