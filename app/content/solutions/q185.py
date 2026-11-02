import numpy as np


def q185(x, mean, components):
    """Project data onto orthonormal PCA components stored as rows, reconstruct it, and
    return mean squared reconstruction error.
    """
    x = np.asarray(x, float)
    centered = x - mean
    reconstructed = centered @ components.T @ components + mean
    return float(np.mean((x - reconstructed) ** 2))
