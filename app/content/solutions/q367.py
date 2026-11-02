import numpy as np


def q367(rewards, epsilon=1e-08):
    """Normalize GRPO reward groups by subtracting each row's mean and dividing by its
    population standard deviation plus epsilon. Constant groups produce zeros.
    """
    r = np.asarray(rewards, float)
    return (r - r.mean(axis=-1, keepdims=True)) / (
        r.std(axis=-1, keepdims=True) + epsilon
    )
