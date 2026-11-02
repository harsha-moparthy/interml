import numpy as np


def q239(fan_in, fan_out, seed=0):
    """Initialize a (fan_in,fan_out) matrix for ReLU networks with normal samples of
    variance 2/fan_in.
    """
    return np.random.default_rng(seed).normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))
