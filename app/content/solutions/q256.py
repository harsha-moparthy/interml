import numpy as np


def q256(gradient, bound):
    """Clip every gradient entry to [-bound,bound], requiring a nonnegative bound."""
    if bound < 0:
        raise ValueError("negative bound")
    return np.clip(gradient, -bound, bound)
