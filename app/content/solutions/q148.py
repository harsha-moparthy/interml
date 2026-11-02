import numpy as np


def q148(x, y, gamma=1):
    """Compute an RBF kernel matrix exp(-gamma*||x_i-y_j||^2) for two feature matrices."""
    x, y = (np.asarray(x, float), np.asarray(y, float))
    distance = (x * x).sum(1)[:, None] + (y * y).sum(1)[None, :] - 2 * x @ y.T
    return np.exp(-gamma * np.maximum(distance, 0))
