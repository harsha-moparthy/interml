import numpy as np


def q028(x, y):
    """Return all pairwise sums of 1-D vectors, with shape (len(x), len(y))."""
    return np.asarray(x)[:, None] + np.asarray(y)[None, :]
