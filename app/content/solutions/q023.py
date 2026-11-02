import numpy as np


def q023(shape, seed=0):
    """Draw standard-normal samples using a local generator; leave global RNG state alone."""
    return np.random.default_rng(seed).standard_normal(shape)
