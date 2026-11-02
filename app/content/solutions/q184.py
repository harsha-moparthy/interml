import numpy as np


def q184(singular_values):
    """Convert PCA singular values to explained-variance ratios. Return all zeros when
    total variance is zero.
    """
    variance = np.asarray(singular_values, float) ** 2
    return variance / variance.sum() if variance.sum() else np.zeros_like(variance)
