import numpy as np


def q035(x, weights, threshold):
    """Project rows through weights and zero rows whose input norm is below threshold."""
    x = np.asarray(x, float)
    return x @ weights * (np.linalg.norm(x, axis=1) >= threshold)[:, None]
