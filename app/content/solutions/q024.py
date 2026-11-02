import numpy as np


def q024(array, shape):
    """Reshape an array and return an independent copy, even when reshape returns a view."""
    return np.asarray(array).reshape(shape).copy()
