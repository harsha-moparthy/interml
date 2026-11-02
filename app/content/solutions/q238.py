import numpy as np


def q238(fan_in, fan_out, seed=0):
    """Initialize a (fan_in,fan_out) matrix with Xavier uniform samples in plus/minus
    sqrt(6/(fan_in+fan_out)).
    """
    bound = np.sqrt(6 / (fan_in + fan_out))
    return np.random.default_rng(seed).uniform(-bound, bound, (fan_in, fan_out))
