import numpy as np


def q125(x):
    """Generate all pairwise products of distinct feature columns, ordered by increasing
    (i,j), i<j. Return (n,0) for fewer than two columns.
    """
    x = np.asarray(x)
    i, j = np.triu_indices(x.shape[1], k=1)
    return x[:, i] * x[:, j]
