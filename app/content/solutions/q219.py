import numpy as np


def q219(x):
    """Compute tanh elementwise with a numerically stable library primitive, preserving
    input array shape.
    """
    return np.tanh(np.asarray(x, float))
