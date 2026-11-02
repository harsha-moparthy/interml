import numpy as np


def q187(y, p):
    """Compute the t-SNE gradient for a symmetric zero-diagonal probability matrix P
    summing to one and low-dimensional coordinates Y. Use Student-t affinities and
    return 4*sum_j((P-Q)*affinity*(y_i-y_j)).
    """
    y = np.asarray(y, float)
    difference = y[:, None] - y[None, :]
    affinity = 1 / (1 + (difference * difference).sum(-1))
    np.fill_diagonal(affinity, 0)
    q = affinity / affinity.sum()
    return 4 * (
        (np.asarray(p) - q)[:, :, None] * affinity[:, :, None] * difference
    ).sum(1)
