import math

import torch


@torch.no_grad()
def initialize_linear(weight, *, method="kaiming", generator=None):
    """Initialize a (fan_out, fan_in) Linear weight in place with normal samples.

    Kaiming uses variance 2/fan_in for ReLU; Xavier uses 2/(fan_in+fan_out)
    for approximately linear/symmetric activations. Initialize biases separately.
    """
    if weight.ndim != 2 or min(weight.shape) < 1:
        raise ValueError("expected a nonempty weight matrix")
    fan_out, fan_in = weight.shape
    if method == "kaiming":
        std = math.sqrt(2 / fan_in)
    elif method == "xavier":
        std = math.sqrt(2 / (fan_in + fan_out))
    else:
        raise ValueError("method must be kaiming or xavier")
    return weight.normal_(0, std, generator=generator)
