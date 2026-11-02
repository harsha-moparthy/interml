import numpy as np


def q033(matrix, axis=0):
    """Return sum, mean, and population standard deviation along the requested axis."""
    x = np.asarray(matrix, float)
    return (x.sum(axis=axis), x.mean(axis=axis), x.std(axis=axis))
